CXX=g++
CPPFLAGS=-pthread -std=c++11 -O3

sim: main.o tAgent.o tHMM.o tGame.o
	$(CXX) $(CPPFLAGS) -o sim $^

clean:
	rm *.o sim
