import jax
import equinox as eqx

def filter_scan(f, init, xs, length=None, reverse=False, unroll=1):
    """
    although simple to implement equinox does not by default include a filter_scan
    function
    see: https://github.com/patrick-kidger/equinox/issues/709
    """

    # Partition the initial carry and sequence inputs into dynamic and static parts
    init_dynamic, init_static = eqx.partition(init, eqx.is_array)
    xs_dynamic, xs_static = eqx.partition(xs, eqx.is_array)

    # Define the scanned function, handling the combination and partitioning
    def scanned_fn(carry_dynamic, x_dynamic):
        # Combine dynamic and static parts for the carry and input
        carry = eqx.combine(carry_dynamic, init_static)
        x = eqx.combine(x_dynamic, xs_static)

        # Apply the original function
        out_carry, out_y = f(carry, x)

        # Partition the outputs into dynamic and static parts
        out_carry_dynamic, out_carry_static = eqx.partition(out_carry, eqx.is_array)
        out_y_dynamic, out_y_static = eqx.partition(out_y, eqx.is_array)

        # Return dynamic outputs and wrap static outputs using Static to prevent tracing
        return out_carry_dynamic, (out_y_dynamic, eqx.internal.Static((out_carry_static, out_y_static)))

    # Use lax.scan with the modified scanned function
    final_carry_dynamic, (ys_dynamic, static_out) = jax.lax.scan(
        scanned_fn, init_dynamic, xs_dynamic, length=length, reverse=reverse, unroll=unroll
    )

    # Extract static outputs
    out_carry_static, ys_static = static_out.value

    # Combine dynamic and static parts of the outputs
    final_carry = eqx.combine(final_carry_dynamic, out_carry_static)
    ys = eqx.combine(ys_dynamic, ys_static)

    return final_carry, ys





