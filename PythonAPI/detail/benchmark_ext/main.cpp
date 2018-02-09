#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/extract.hpp>
#include <iostream>
#include <string.h>
#include <cstring>
#include "Matrix.hh"
#include "csa.hh"
#include "match.hh"

boost::python::tuple benchmark(boost::python::list& map1, boost::python::list& map2, boost::python::list& dims, boost::python::list& dc){

	int rows = boost::python::extract<int>(dims[0]);
	int cols = boost::python::extract<int>(dims[1]);

	double* bmap1 = new double[rows * cols]; 
	double* bmap2 = new double[rows * cols];

        for(int i = 0; i<rows * cols;i++){
		bmap1[i] = boost::python::extract<double>(map1[i]);
		bmap2[i] = boost::python::extract<double>(map2[i]);
	};

        double maxDist = boost::python::extract<double>(dc[0]);
	double outlierCost = boost::python::extract<double>(dc[1]);

	const double idiag = sqrt( rows*rows + cols*cols );
	const double oc = outlierCost*maxDist*idiag;

	Matrix m1, m2;
	const double cost = matchEdgeMaps(
	        Matrix(rows,cols,bmap1), Matrix(rows,cols,bmap2),
	        maxDist*idiag, oc,
	        m1, m2);

	boost::python::list match1;
	boost::python::list match2;
        for(int i = 0; i<rows*cols;i++) match1.append(m1.data()[i]);
        for(int i = 0; i<rows*cols;i++) match2.append(m2.data()[i]);
        delete bmap1;
        delete bmap2;
	return boost::python::make_tuple(match1, match2, cost);
}
