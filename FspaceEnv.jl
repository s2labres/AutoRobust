export FspaceEnv

struct FspaceEnvParams{T}
    accuracy::T
    statesize:T
    embeddingsize:Int
    max_steps::Int
end

Base.show(io::IO, params::FspaceEnvParams) = print(
    io,
    join(["$p=$(getfield(params, p))" for p in fieldnames(FspaceEnvParams)], ","),
)

function FspaceEnvParams{T}(;
    accuracy=0.7,
    statesize=1,
    embeddingsize=10,
    max_steps=1e5,
) where {T}
    FspaceEnvParams{T}(
        accuracy,
        statesize,
        embeddingsize,
        max_steps,
    )
end

mutable struct FspaceEnv{T,ACT} <: AbstractEnv
    params::FspaceEnvParams{T}
    state::Vector{T}
    action::ACT
    done::Bool
    t::Int
    rng::AbstractRNG
end

"""
    FspaceEnv(;kwargs...)

# Keyword arguments
- `T = Float64`
- `continuous = false`
- `rng = Random.GLOBAL_RNG`
- `accuracy = T(0.7)`
"""
function FspaceEnv(; T=Float64, continuous=false, rng=Random.GLOBAL_RNG, kwargs...)
    params = FspaceEnvParams{T}(; kwargs...)
    env = FspaceEnv(params, zeros(T, 4), continuous ? zero(T) : zero(Int), false, 0, rng)
    reset!(env)
    env
end

FspaceEnv{T}(; kwargs...) where {T} = FspaceEnv(T=T, kwargs...)

Random.seed!(env::FspaceEnv, seed) = Random.seed!(env.rng, seed)
RLBase.reward(env::FspaceEnv{T}) where {T} = env.done ? zero(T) : one(T)
RLBase.is_terminated(env::FspaceEnv) = env.done
RLBase.state(env::FspaceEnv) = env.state

function RLBase.state_space(env::FspaceEnv{T}) where {T}
    ((-2 * env.params.spacesize) .. (2 * env.params.spacesize)) ×
    (typemin(T) .. typemax(T)) ×
    ((-2 * env.params.embeddingsize) .. (2 * env.params.embeddingsize)) ×
    (typemin(T) .. typemax(T))
end

RLBase.action_space(env::FspaceEnv{<:AbstractFloat,Int}, player) = Base.OneTo(2)
RLBase.action_space(env::FspaceEnv{<:AbstractFloat,<:AbstractFloat}, player) = -1.0 .. 1.0

function RLBase.reset!(env::CartPoleEnv{T}) where {T}
    env.state[:] = T(0.1) * rand(env.rng, T, 4) .- T(0.05)
    env.t = 0
    env.action = rand(env.rng, action_space(env))
    env.done = false
    nothing
end

function (env::CartPoleEnv)(a::AbstractFloat)
    @assert a in action_space(env)
    env.action = a
    _step!(env, a)
end

function (env::CartPoleEnv)(a::Int)
    @assert a in action_space(env)
    env.action = a
    _step!(env, a == 2 ? 1 : -1)
end

function _step!(env::CartPoleEnv, a)
    env.t += 1
    force = a * env.params.forcemag
    x, xdot, theta, thetadot = env.state
    costheta = cos(theta)
    sintheta = sin(theta)
    tmp = (force + env.params.polemasslength * thetadot^2 * sintheta) / env.params.totalmass
    thetaacc =
        (env.params.gravity * sintheta - costheta * tmp) / (
            env.params.halflength *
            (4 / 3 - env.params.masspole * costheta^2 / env.params.totalmass)
        )
    xacc = tmp - env.params.polemasslength * thetaacc * costheta / env.params.totalmass
    env.state[1] += env.params.dt * xdot
    env.state[2] += env.params.dt * xacc
    env.state[3] += env.params.dt * thetadot
    env.state[4] += env.params.dt * thetaacc
    env.done =
        abs(env.state[1]) > env.params.xthreshold ||
        abs(env.state[3]) > env.params.thetathreshold ||
        env.t > env.params.max_steps
    nothing
end