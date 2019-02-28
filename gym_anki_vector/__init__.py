from gym.envs.registration import register

register(
    id='anki-vector-v0',
    entry_point='gym_anki_vector.envs:AnkiVectorEnv',
)
