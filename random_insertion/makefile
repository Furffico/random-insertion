cxx = g++
pythonversion = $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
pythonroot = $(shell "python$(pythonversion)" -c "import sys; print(sys.prefix)")
numpyroot = $(shell "python$(pythonversion)" -c "import numpy,os;\
	print(os.path.join(os.path.dirname(numpy.__file__), '_core' if numpy.__version__>='2.0' else 'core', 'include'))")
cxxflags = -std=c++17 -O3 -Os -fPIC -I$(pythonroot)/include/python$(pythonversion) -I$(numpyroot)

sources = $(wildcard src/*.cpp)
targets := $(patsubst %.cpp,%.o,$(sources))
depends := $(patsubst %.cpp,%.d,$(sources))

.PHONY: default clean;

default: _core.so;

-include $(depends)

$(targets): %.o: %.cpp
	$(cxx) $(cxxflags) -MMD -MP -c -o $@ $<

_core.so: $(targets)
	$(cxx) $(cxxflags) --shared -o ./_core.so $^

clean:
	rm -f src/*.o src/*.d *.o *.d *.so
