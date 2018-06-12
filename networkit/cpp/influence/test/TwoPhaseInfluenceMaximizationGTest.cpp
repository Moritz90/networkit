#ifndef NOGTEST

#include "TwoPhaseInfluenceMaximizationGTest.h"

#include "../TwoPhaseInfluenceMaximization.h"
#include "../../auxiliary/Log.h"
#include "../../auxiliary/StringBuilder.h"
#include "../../graph/GraphBuilder.h"

#include <cmath>

namespace NetworKit {

TEST_F(TwoPhaseInfluenceMaximizationGTest, twoStarsGraph) {
    // graph consisting of two stars with 4 rays each; all edge weights are 1 (-> deterministic propagation)
    auto builder = GraphBuilder(10, true, true);
    builder.addHalfEdge(0, 1, 1.0);
    builder.addHalfEdge(0, 2, 1.0);
    builder.addHalfEdge(0, 3, 1.0);
    builder.addHalfEdge(0, 4, 1.0);
    builder.addHalfEdge(5, 6, 1.0);
    builder.addHalfEdge(5, 7, 1.0);
    builder.addHalfEdge(5, 8, 1.0);
    builder.addHalfEdge(5, 9, 1.0);
    const auto G = builder.toGraph(true);

    INFO("G: " , G.toString());

    const auto expected_result = std::unordered_set<node>{0, 5};
    // stochastic algorithm, so let's run multiple times
    count correct_tries = 0;
    for (count i = 0; i < 10; ++i) {
        TwoPhaseInfluenceMaximization tim{G, 2, 0.5, 1.0};
        tim.run();
        auto result = tim.topKInfluencers();
        if (result == expected_result) {
            correct_tries++;
        } else {
            INFO(std::string{"Mismatch detected. Actual result: "} + Aux::toString(result));
        }
    }

    EXPECT_GE(correct_tries, 9) << "algorithm failed to produce optimal result more than 90% of the time";
}

TEST_F(TwoPhaseInfluenceMaximizationGTest, exampleGraph) {
    // example graph from Tang's paper (slightly modified)
    auto builder = GraphBuilder(4, true, true);
    builder.addHalfEdge(3, 0, 0.8);
    builder.addHalfEdge(2, 3, 0.1);
    builder.addHalfEdge(2, 0, 0.1);
    builder.addHalfEdge(1, 3, 0.1);
    builder.addHalfEdge(1, 0, 0.1);
    const auto G = builder.toGraph(true);

    INFO("G: " , G.toString());

    const auto expected_result = std::unordered_set<node>{3};
    count correct_tries = 0;
    for (count i = 0; i < 10; ++i) {
        TwoPhaseInfluenceMaximization tim{G, 1, 0.1, 1.0};
        tim.run();
        auto result = tim.topKInfluencers();
        if (result == expected_result) {
            correct_tries++;
        } else {
            INFO(std::string{"Mismatch detected. Actual result: "} + Aux::toString(result));
        }
    }

    EXPECT_GE(correct_tries, 9) << "algorithm failed to produce optimal result more than 90% of the time";
}

} /* namespace NetworKit */

#endif /*NOGTEST */