import numpy as np


def modified_doppler_effect(freq, obs_pos, obs_vel, obs_speed, src_pos,
                            src_vel, src_speed, sound_vel):
    # Normalize velocity vectors to find their directions (zero values
    # have no direction).
    if not np.all(src_vel == 0):
        src_vel = src_vel / np.linalg.norm(src_vel)
    if not np.all(obs_vel == 0):
        obs_vel = obs_vel / np.linalg.norm(obs_vel)

    src_to_obs = obs_pos - src_pos
    obs_to_src = src_pos - obs_pos
    if not np.all(src_to_obs == 0):
        src_to_obs = src_to_obs / np.linalg.norm(src_to_obs)
    if not np.all(obs_to_src == 0):
        obs_to_src = obs_to_src / np.linalg.norm(obs_to_src)

    src_radial_vel = src_speed * src_vel.dot(src_to_obs)
    obs_radial_vel = obs_speed * obs_vel.dot(obs_to_src)

    fp = ((sound_vel + obs_radial_vel) / (sound_vel - src_radial_vel)) * freq

    return fp


def inverse_square_law_observer_receiver(obs_pos, src_pos, K=1.0, eps=0.0):
    """
    Computes the inverse square law for an observer receiver pair.
    Follows https://en.wikipedia.org/wiki/Inverse-square_law
    """
    distance = np.linalg.norm(obs_pos - src_pos)
    return K * 1.0 / (distance**2 + eps)
