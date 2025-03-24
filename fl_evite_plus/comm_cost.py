"""
Communication cost module.
Computes communication cost as energy per bit times distance squared.
"""

def energy_comm_cost(params, energy_per_bit, distance):
    """
    Compute the communication cost.

    Args:
        params: List of numpy arrays representing model parameters.
        energy_per_bit: Energy consumption per bit (configurable).
        distance: Communication distance (configurable).

    Returns:
        The cost computed as:
            energy_per_bit * (distance ** 2) * total_bits,
        where total_bits is the total number of bits (assuming 32 bits per parameter value).
    """
    # Calculate total bits assuming each parameter is a 32-bit float.
    total_bits = sum(p.size for p in params) * 32
    return energy_per_bit * (distance ** 2) * total_bits
