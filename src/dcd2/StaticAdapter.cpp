/*
 * StaticAdapter.cpp
 *
 *  Created on: 10.01.2014
 *      Author: cls
 */

#include "StaticAdapter.h"

namespace NetworKit {

StaticAdapter::StaticAdapter(Clusterer* algo) : algo(algo) {

}

void StaticAdapter::update(std::vector<GraphEvent>& stream) {
	// do nothing
}

Partition StaticAdapter::detect() {
	return algo->run(*G);
}

} /* namespace NetworKit */