from argparse import Namespace
from typing import TypeAlias, Sequence
from warnings import warn

from distrax import Distribution
import flax
import flax.struct as struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from ..buffer.ppo_buffer import PPOAgentState as AgentState, PPOExperience as Experience
from ..env import ObservationSpace, ActionSpace, get_example_observation
from ..model.actor_critic import Actor, Critic
from ..model.models import NNConfig
from ..utils import global_norm, split_into_minibatch


PPOTrainState: TypeAlias = tuple[TrainState, TrainState]

class PPOConfig(struct.PyTreeNode):
    pi_lr: float
    val_lr: float
    entropy_coef: float
    ratio_clip: float
    value_clip: float
    grad_clip_norm: float
    # for lr scheduling
    gradient_update_steps: int
    end_pi_lr: float = 0
    end_val_lr: float = 0
    @classmethod
    def init(cls, args: Namespace):
        if args.algo == 'ippo':
            gradient_update_steps = args.total_env_steps // args.num_envs * args.ppo_epochs * args.num_minibatches
            return cls(
                pi_lr=args.pi_lr,
                val_lr=args.val_lr,
                entropy_coef=args.entropy_coef,
                ratio_clip=args.ratio_clip,
                value_clip=args.value_clip,
                grad_clip_norm=args.grad_clip_norm,
                gradient_update_steps=gradient_update_steps
            )
        else:
            raise NotImplementedError(f"algorithm {args.algo} does not have a config initialization funciton")

class PPOAgent(struct.PyTreeNode):
    actor_fn: Actor = struct.field(pytree_node=False)
    critic_fn: Critic = struct.field(pytree_node=False)
    agent_name: str = struct.field(pytree_node=False)
    config: PPOConfig
    obs_space: ObservationSpace
    chunk_length: int
    num_minibatches: int
    minibatch_num_chunks: int | None = struct.field(pytree_node=False)
    @classmethod
    def init(cls, agent_name: str, args: Namespace, obs_space: ObservationSpace, action_space: ActionSpace, model_config: NNConfig):
        print("init", agent_name, args.use_rnn)
        actor_fn = Actor(
            nn_configs=model_config,
            obs_space=obs_space,
            action_space=action_space,
            mlp_hidden_layers=args.mlp_hidden_layer,
            mlp_hidden_size=args.mlp_hidden_size,
            use_rnn=args.use_rnn,
            rnn_hidden_layers=args.rnn_hidden_layers,
            rnn_hidden_size=args.rnn_hidden_size,
            use_rnn_layer_norm=not args.no_rnn_layer_norm,
            use_embedding_layer_norm=not args.no_embedding_layer_norm,
            use_final_layer_norm=args.use_final_layer_norm
        )
        critic_fn = Critic(
            nn_configs=model_config,
            obs_space=obs_space,
            mlp_hidden_layers=args.mlp_hidden_layer,
            mlp_hidden_size=args.mlp_hidden_size,
            use_rnn=args.use_rnn,
            rnn_hidden_layers=args.rnn_hidden_layers,
            rnn_hidden_size=args.rnn_hidden_size,
            use_rnn_layer_norm=not args.no_rnn_layer_norm,
            use_embedding_layer_norm=not args.no_embedding_layer_norm,
            use_final_layer_norm=args.use_final_layer_norm
        )

        if args.batch_size is None:
            minibatch_num_chunks = None 
        else:
            minibatch_num_chunks = args.batch_size // args.chunk_length
            if args.batch_size % args.chunk_length != 0:
                warn(
                    f"Batch size {args.batch_size} is not a multiple of chunk length {args.chunk_length}. "
                    f"It will be adjusted to {minibatch_num_chunks * args.chunk_length} to fit evenly."
                )

        if args.episode_length % args.chunk_length != 0:
            warn(
                f"Training episode length {args.episode_length} is not a multiple of chunk length {args.chunk_length}. "
                f"It will be adjusted to {args.episode_length // args.chunk_length * args.chunk_length} during data preparation for model updates."
            )

        if minibatch_num_chunks is None:
            total_num_chunks = args.episode_length // args.chunk_length * args.num_envs
            if total_num_chunks % args.num_minibatches != 0:
                warn(
                    f"The estimated number of chunks ({total_num_chunks}) based on the total number of environments is not divisible by the number of minibatches ({args.num_minibatches}). "
                    f"Some data may not be used during model updates."
                )
        else:
            total_num_chunks = args.episode_length // args.chunk_length * args.num_envs
            if total_num_chunks % minibatch_num_chunks != 0:
                warn(
                    f"The estimated number of chunks ({total_num_chunks}) based on the total number of environments is not divisible by the number of chunks per minibatch ({minibatch_num_chunks}). "
                    f"Some data may not be used during model updates."
                )

        return cls(
            actor_fn=actor_fn,
            critic_fn=critic_fn,
            agent_name=agent_name,
            config=PPOConfig.init(args),
            obs_space=obs_space,
            chunk_length=args.chunk_length,
            num_minibatches=args.num_minibatches,
            minibatch_num_chunks=minibatch_num_chunks
        )
    
    def agent_state_init(self, batch_shape: Sequence[int]) -> AgentState:
        # print('batch_shape', batch_shape)
        actor_rnn_state = self.actor_fn.default_rnn_state(batch_shape)
        critic_rnn_state = self.critic_fn.default_rnn_state(batch_shape)
        return AgentState(actor_rnn_state, critic_rnn_state)
    
    def agent_state_reset(self, agent_state: AgentState, done: jax.Array) -> AgentState:
        """Params:
            agent_state.***_rnn_states: [*batch_shape, rnn_state_size]
            done: [*batch_shape]
        """
        default_agent_state = self.agent_state_init(agent_state.get_batch_shape())
        done = jnp.expand_dims(done, -1)
        return jax.tree_util.tree_map(
            lambda x, y: x * (1 - done) + y, agent_state, default_agent_state
        )

    def train_state_init(self, rng: jax.Array) -> PPOTrainState:
        example_rnn_state = self.agent_state_init(batch_shape=(1,))
        rng, actor_rng = jax.random.split(rng)
        actor_params = self.actor_fn.init(
            actor_rng,
            example_rnn_state.actor_rnn_state,
            get_example_observation(batch_shape=(1,), obs_space=self.obs_space)
        )
        actor_tx = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip_norm),
            optax.adam(
                learning_rate=optax.linear_schedule(
                    init_value=self.config.pi_lr,
                    end_value=self.config.end_pi_lr,
                    transition_steps=self.config.gradient_update_steps
                )
            )
        )
        actor_train_state = TrainState.create(
            apply_fn=self.actor_fn.apply,
            params=actor_params,
            tx=actor_tx
        )

        rng, critic_rng = jax.random.split(rng)
        critic_params = self.critic_fn.init(critic_rng, example_rnn_state.critic_rnn_state, {'obs': jnp.zeros((1, *self.obs_space['obs']))})
        critic_tx = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip_norm),
            optax.adam(
                learning_rate=optax.linear_schedule(
                    init_value=self.config.val_lr,
                    end_value=self.config.end_val_lr,
                    transition_steps=self.config.gradient_update_steps
                )
            )
        )
        critic_train_state = TrainState.create(
            apply_fn=self.critic_fn.apply,
            params=critic_params,
            tx=critic_tx
        )

        return (actor_train_state, critic_train_state)
    
    @jax.jit
    def rollout_step(
        self,
        rng: jax.Array,
        train_state: PPOTrainState,
        agent_state: AgentState,
        obs: jax.Array
    ) -> tuple[AgentState, jax.Array, jax.Array, jax.Array]:
        actor_rnn_state, critic_rnn_state = agent_state.actor_rnn_state, agent_state.critic_rnn_state
        actor_train_state, critic_train_state = train_state

        obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), obs)
        next_actor_rnn_state: jax.Array; pi: Distribution
        next_actor_rnn_state, pi = self.actor_fn.apply(actor_train_state.params, actor_rnn_state, obs)
        next_critic_rnn_state: jax.Array; val: jax.Array
        next_critic_rnn_state, val = self.critic_fn.apply(critic_train_state.params, critic_rnn_state, obs)
        action: jax.Array = pi.sample(seed=rng)
        log_p = pi.log_prob(action)
        if len(log_p.shape) > 1:
            log_p = jnp.sum(log_p, axis=1)
        action, log_p, val = action.squeeze(0), log_p.squeeze(0), val.squeeze(0)

        return AgentState(next_actor_rnn_state, next_critic_rnn_state), action, log_p, val
    
    @jax.jit
    def get_val(
        self,
        train_state: PPOTrainState,
        agent_state: AgentState,
        obs: jax.Array
    ) -> jax.Array:        
        critic_rnn_state = agent_state.critic_rnn_state
        actor_train_state, critic_train_state = train_state
        obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), obs)
        val: jax.Array
        next_critic_rnn_state, val = self.critic_fn.apply(critic_train_state.params, critic_rnn_state, obs)
        return val.squeeze(0)
    
    def actor_loss_fn(self, actor_params: flax.core.FrozenDict, buffer: Experience, advantage: jax.Array):
        """Notation:
            L: chunk_length | epsiode_length
            B: batch_size
            A: observation_space
        Input:
            buffer: [L, B, *shape]
            adv: [L, B]
        """
        pi: Distribution
        rnn_state, pi = self.actor_fn.apply(actor_params, buffer.agent_state.actor_rnn_state[0], buffer.obs) # [L, B, *A]
        # print(self.agent_name, buffer.action[self.agent_name])
        # import pdb; pdb.set_trace()
        log_p = pi.log_prob(buffer.action[self.agent_name]) # [L, B]
        if len(log_p.shape) > 2:
            log_p = jnp.sum(log_p, axis=2)
        ratio = jnp.exp(log_p - buffer.log_p) # [L, B]

        unclipped_pi_obj = ratio * advantage # [L, B]
        clipped_pi_obj = jnp.clip(ratio, 1 - self.config.ratio_clip, 1 + self.config.ratio_clip) * advantage # [L, B]

        pi_loss = -jnp.minimum(unclipped_pi_obj, clipped_pi_obj) # [L, B]
        pi_loss = pi_loss.mean()
        entropy = pi.entropy() # [L, B]
        entropy = entropy.mean()

        loss = pi_loss - self.config.entropy_coef * entropy
        return loss, {
            'pi_loss': pi_loss,
            'entropy': entropy,
            'log_p': log_p.mean(),
            'ratio_clip_fraction': (jnp.abs(ratio - 1) > self.config.ratio_clip).mean()
        }

    def critic_loss_fn(self, critic_params: flax.core.FrozenDict, buffer: Experience, target_val: jax.Array):
        """Notation:
            L: chunk_length | epsiode_length
            B: batch_size
        Input:
            buffer: [L, B, *shape]
            target_val: [L, B]
        """
        val: jax.Array
        rnn_state, val = self.critic_fn.apply(critic_params, buffer.agent_state.critic_rnn_state[0], buffer.obs) # [L, B]

        unclipped_val_loss = 0.5 * jnp.square(val - target_val) # [L, B]
        clipped_val = target_val + (val - target_val).clip(-self.config.value_clip, self.config.value_clip) # [L, B]
        clipped_val_loss = 0.5 * jnp.square(clipped_val - target_val) # [L, B]

        val_loss = jnp.maximum(unclipped_val_loss, clipped_val_loss).mean()
        return val_loss, {
            'val_loss': val_loss,
            'val_clip_fraction': (jnp.abs(val - target_val) > self.config.value_clip).mean()
        }
    
    def model_gradient_update(
        self,
        train_state: PPOTrainState,
        data: tuple[Experience, jax.Array, jax.Array]
    ) -> tuple[PPOTrainState, dict]:
        buffer, advantage, target_val = data
        actor_train_state, critic_train_state = train_state

        (actor_loss, actor_aux), actor_grads = jax.value_and_grad(self.actor_loss_fn, has_aux=True)(
            actor_train_state.params,
            buffer=buffer,
            advantage=advantage
        )
        actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)

        (critic_loss, critic_aux), critic_grads = jax.value_and_grad(self.critic_loss_fn, has_aux=True)(
            critic_train_state.params,
            buffer=buffer,
            target_val=target_val
        )
        critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
        
        optim_log = {
            'actor_loss': actor_loss,
            'actor_grad_norm': global_norm(actor_grads),
            'critic_loss': critic_loss,
            'critic_grad_norm': global_norm(critic_grads)
        }
        optim_log.update(actor_aux)
        optim_log.update(critic_aux)
        return (actor_train_state, critic_train_state), optim_log
    
    def learn(
        self,
        rng: jax.Array,
        train_state: PPOTrainState,
        buffer: Experience,
        advantage: jax.Array,
        target_val: jax.Array
    ) -> tuple[PPOTrainState, dict]:
        # Use the same rng to ensure the random permutated indicies are the same for buffer/advanrage/target_val
        rng, batch_split_rng = jax.random.split(rng)
        buffer_batch = split_into_minibatch(
            batch_split_rng,
            buffer,
            chunk_length=self.chunk_length,
            num_minibatches=self.num_minibatches,
            minibatch_num_chunks=self.minibatch_num_chunks
        )
        advantage_batch = split_into_minibatch(
            batch_split_rng,
            advantage,
            chunk_length=self.chunk_length,
            num_minibatches=self.num_minibatches,
            minibatch_num_chunks=self.minibatch_num_chunks
        )
        target_val_batch = split_into_minibatch(
            batch_split_rng,
            target_val,
            chunk_length=self.chunk_length,
            num_minibatches=self.num_minibatches,
            minibatch_num_chunks=self.minibatch_num_chunks
        )
        train_state, optim_log = jax.lax.scan(self.model_gradient_update, train_state, (buffer_batch, advantage_batch, target_val_batch))
        optim_log = jax.tree_util.tree_map(jnp.mean, optim_log)
        return train_state, optim_log
    
    def learn2(
        self,
        train_state: PPOTrainState,
        data
    ) -> PPOTrainState:
        buffer, advantage, target_val = data
        train_state, optim_log = self.model_gradient_update(train_state, (buffer, advantage, target_val))
        return train_state, None
