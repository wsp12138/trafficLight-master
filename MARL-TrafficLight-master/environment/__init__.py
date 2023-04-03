from gymnasium.envs.registration import register

#把自己的环境注册到gym里
register(
    id='sumo-rl-v0',
    entry_point='sumo_rl.environment.env:SumoEnvironment',
    kwargs={'single_agent': True},
)
