# How to compile benchmark_ext.so
1. You need to have boost with boost/python installed. You will also need bjam (this one should also go with your boost distribution).
2. Go to Jamroot. Change use-project-boost so that it points at your boost distribution.
3. While being in the same directory in which this readme is type on your terminal "bjam ."
4. Under "bin/gcc-5.4.0/debug/" (or other gcc version) you have your own just compiled benchmark_ext.so
