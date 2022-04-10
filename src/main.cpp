//
// Created by Conor Stevenson on 03/04/2022.
//
#include <vector>
#include <algorithm>

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/utility/source.hpp>

#define N 4


namespace compute = boost::compute;

const char sparse_vect_mult_source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void SparseVectMult(global float* COOmat)
        {

        }
);

int main()
{
    compute::device device = compute::system::default_device();
    compute::context context(device);

    std::vector<float> a = { 1, 2, 3, 4 };

    compute::buffer buffer_a(context, 4 * sizeof(float));

    compute::program program =
            compute::program::create_with_source(sparse_vect_mult_source, context);
    program.build();

    compute::kernel kernel(program, "SparseVectMult");

    kernel.set_arg(0, buffer_a);

    compute::command_queue queue(context, device);

    queue.enqueue_write_buffer(buffer_a, 0, N * sizeof(float), a.data());

    queue.enqueue_1d_range_kernel(kernel, 0, N, 0);

    return 0;
}

