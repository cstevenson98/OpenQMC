//
// Created by conor on 18/04/22.
//

#define BOOST_TEST_MODULE la_test
#include <boost/test/included/unit_test.hpp>

#include "../../la/SparseELL.cuh"
#include "../../la/Dense.cuh"

using t_stdMat = vector<vector<complex<double>>>;


BOOST_AUTO_TEST_SUITE( sparse_test )

BOOST_AUTO_TEST_CASE( sparse_add_test )
{

    t_stdMat base1Data = {{ 2, 0, 0},
                          { 0, 1, 0}};
    t_stdMat base2Data = {{ 1, 0, 1},
                          { 1, 1, 0}};
    t_stdMat base3Data = {{-2, 0, 0},
                          {-1, 0, 0}};

    Dense base1(2, 3), base2(2, 3), base3(2, 3);

    base1.Data = base1Data;
    base2.Data = base2Data;
    base3.Data = base3Data;

    //1
    Sparse S1 = ToSparseCOO(base1);
    Sparse S2 = ToSparseCOO(base2);
    Sparse S3 = ToSparseCOO(base3);

}

BOOST_AUTO_TEST_SUITE_END()
