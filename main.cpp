#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <map>
#include <math.h>
#include <time.h>
#include <iostream>
#include <thread>
#include <functional> // for ref
#include "globalConst.h"
#include "tHMM.h"
#include "tAgent.h"
#include "tGame.h"
#include "params/params.h"

#ifdef _WIN32
	#include <process.h>
#else
	#include <unistd.h>
#endif

#define randDouble ((double)rand()/(double)RAND_MAX)

using namespace std;
using namespace Params;

double perSiteMutationRate=0.005;
int update=0;
size_t repeats=1;
int maxAgent=100;
int totalGenerations=200;
const int mp=1;
int gameID=0;

enum class Selection : int {
		task = 1,
		phi = 2,
		r = 4,
		diameter = 8,
		genome = 16,
		none = 32,
      sparseness = 64, // sparseness and connectedness are inverses of each other
      connectedness = 128,
      predIs2s = 256,
      predIs2m = 512 };

map<string, Selection> SelectionValues{
 make_pair("task",Selection::task),
 make_pair("phi",Selection::phi),
 make_pair("r",Selection::r),
 make_pair("diameter",Selection::diameter),
 make_pair("genome",Selection::genome),
 make_pair("none",Selection::none),
 make_pair("sparseness",Selection::sparseness),
 make_pair("connectedness",Selection::connectedness),
 make_pair("predictiveISensorsToSensors",Selection::predIs2s),
 make_pair("predictiveISensorsToMotors",Selection::predIs2m)
};

bool isaSelectionValue(string value) {
	if (SelectionValues.find(value) == SelectionValues.end())
		return false;
	else
		return true;
}

void computeLOD(FILE *f,FILE *g, tAgent *agent,tGame *game);
const char* cstr(string s) { return s.c_str(); }

// processes a subsection of the population
void threadedEvaluateFitness(int chunkBegin, int chunkEnd, const vector<tAgent*>& agents, tGame& game, int& evaluations) {
	for (int chunk_index=chunkBegin; chunk_index<chunkEnd; chunk_index++) {
		for (int replicate_i=0; replicate_i < evaluations; replicate_i++) {
         switch(gameID) {
            case 0: // block catcher
            game.executeGame(agents[chunk_index], 2, nullptr, true, -1, -1);
               break;
            case 1: // random generator
            game.executeGame2(agents[chunk_index]);
               break;
            default: // same as 0
            game.executeGame(agents[chunk_index], 2, nullptr, true, -1, -1);
               break;
         }
			agents[chunk_index]->fitnesses.push_back(agents[chunk_index]->fitness);
		}
	}
}

void show(vector<vector<int>>& v) {
   size_t x=0,y=0;
   cout << endl;
   for (x=0; x<v.size(); x++) {
      for (y=0; y<v[x].size(); y++) {
         cout << v[x][y] << "  ";
      }
      cout << endl;
   }
}
int main(int argc, char *argv[]) {
	string experimentID;
	int replicateID=0;
	vector<tAgent*>agent;
	vector<tAgent*>nextGen;
	int who=0;
	size_t i;
	tAgent *masterAgent=nullptr;
	tGame *game=nullptr;
	FILE *LODFile=nullptr;
	FILE *genomeFile=nullptr;
	bool showhelp;
	float regimeValueLimit; // evolve until value of fitness is reached, then drop into basic task fitness
	float maxFitness; // holds max fitness of population, used for proportional selection
	float tempFitness; // holds max/actual fitness while being geometrically accumulated
	int regimeGenLimit; // evolve until evolveGenLimit generations passed, then drop into basic task fitness
	vector<string> selectionOn;
	int startGenes;
	bool stopOnLimit;
	int selectionRegime = 0; // bitmask for how to perform selection (fitness, phi, r, diameter, genome)
	int nregimes = 0;
	int evaluations=0;						// how many times to test evaluating fitness of an agent
	string filenameLOD, filenameGenome, filenameStartWith;
	int nthreads=0;
   //int exportLODgenomes=0;
	vector<thread> threads;

	addp(TYPE::BOOL, &showhelp);
	addp(TYPE::STRING, &filenameLOD, "--lod", "filename to save Line of Descent.");
	addp(TYPE::STRING, &experimentID, "--experiment", "unique identifier for this experiment, shared by all replicates.");
	addp(TYPE::INT, &replicateID, "--replicate", "unique number to identify this replicate in this experiment.");
	addp(TYPE::STRING, &filenameGenome, "--genome", "filename to save genomes of the LODFile.");
	addp(TYPE::INT, &totalGenerations, "200", false, "--generations", "number of generations to simulate (updates).");
	addp(TYPE::STRING, &filenameStartWith, "none", false, "--startWith", "specify a genome file used to seed the population.");
	addp(TYPE::BOOL, &stopOnLimit, "false", false, "--stopOnLimit", "if a limit is specified, then the simulation will stop at the limit.");
	addp(TYPE::INT, &startGenes, "5000", false, "--startGenes", "number of genes with which first organisms begin.");
	addp(TYPE::INT, &evaluations, "1", false, "--evaluations", "number of evaluations for an agent's fitness evaluation.");
	addp(TYPE::INT, &nthreads, "1", false, "--nthreads", (string("number of threads to use. This system reports ")+to_string(thread::hardware_concurrency())+string(" cores available.")).c_str());
	addp(TYPE::STRING, &selectionOn, -1, false, "--selectionOn", "list of parameters on which to perform selection. Valid options are: task, phi, r, diameter, genome, none.");
	addp(TYPE::INT, &regimeGenLimit, "-1", false, "--regimeGenLimit", "generation limit to use in all regimes before going back to task fitness.");
	addp(TYPE::FLOAT, &regimeValueLimit, "-1", false, "--regimeValueLimit", "value limit to use in all regimes before going back to task fitness.");
   //addp(TYPE::INT, &exportLODgenomes, "-1", false, "--exportLODgenomes", "number of generations between export snapshots of the genomes along LOD.");
   addp(TYPE::INT, &gameID, "0", false, "--gameID", "which game to use? {0:catcher, 1:randomness}");
	argparse(argv);
	if (showhelp) {
		cout << argdetails() << endl;
		cout << "Example minimal invocation:" << endl;
		cout << argv[0] << " --experiment=linearFitness --replicate=1 --lod=lineOfDescent.lod --genome=genome.gen" << endl;
		cout << "or" << endl;
		cout << argv[0] << " --experiment linearFitness --replicate 1 --lod lineOfDescent.lod --genome genome.gen" << endl;
		cout << endl;
		exit(0);
	}
	
	/// inputs for selectionRegime from --selectionOn
	if (selectionOn.size() == 0) {
		selectionOn.push_back("task");
		//cout << "no regimes specified for selection after --selectionOn option." << endl;
		//cout << endl;
		//exit(1);
	}
	for (string& argument : selectionOn) {
		if (isaSelectionValue(argument)) {
			selectionRegime |= (int)SelectionValues[argument];
		} else {
			cout << "invalid --selectionOn option: '" << argument << "'" << endl;
			cout << endl;
			exit(1);
		}
	}

	/// count number of (unique & valid) regimes specified
	if ((selectionRegime & (int)Selection::none) == (int)Selection::none) {
		nregimes = 1;
		selectionRegime = (int)Selection::none;
	} else {
		nregimes = 0;
		int regimeBitmask = selectionRegime;
		for (nregimes = 0; regimeBitmask; nregimes++) { 
			regimeBitmask &= regimeBitmask - 1;
		}
	}

    srand(getpid());
    masterAgent=new tAgent();

	LODFile=fopen(cstr(filenameLOD),"w+t");
	genomeFile=fopen(cstr(filenameGenome),"w+t");	
	srand(getpid());
	agent.resize(maxAgent);
	masterAgent=new tAgent;
	vector<vector<int> > data;
	game=new tGame;
	masterAgent->setupRandomAgent(startGenes);
	
	masterAgent->setupPhenotype();

	if (filenameStartWith != "none") {
		masterAgent->loadAgent(filenameStartWith.c_str());
		for(i=0;i<agent.size();i++){
			agent[i]=new tAgent;
			agent[i]->inherit(masterAgent,0.0,0); // small mutation rate to preserve genome
		}
	} else {
		for(i=0;i<agent.size();i++){
			agent[i]=new tAgent;
			agent[i]->inherit(masterAgent,0.5,0); // large mutation rate to spread the genotypic population
		}
	}
	nextGen.resize(agent.size());
	masterAgent->nrPointingAtMe--;
	cout<<"setup complete"<<endl;
 printf("%s %s %s %s %s %s %s %s %s %s %s\n", "update","(double)maxFitness","maxPhi","r", "maxDiameter", "connectedness", "sparseness","maxPIs2s","maxPIs2m", "agent[who]->correct","agent[who]->incorrect");

	while(update<totalGenerations) {
		for(i=0;i<agent.size();i++) {
			agent[i]->fitness=0.0;
			agent[i]->fitnesses.clear();
		}

		/// perform fitness evaluation if not none selection specified
		if ((selectionRegime & (int)Selection::none) != (int)Selection::none) {
			threads.clear();
			int chunksize=agent.size()/nthreads;
			for (int threadid=0; threadid < nthreads; threadid++)
				threads.push_back(thread(threadedEvaluateFitness, chunksize*threadid, chunksize*threadid+chunksize, ref(agent), ref(*game), ref(evaluations)));
			if (agent.size()%nthreads != 0) // take care of any uneven division of workload
			{
				threads.push_back(thread(threadedEvaluateFitness, nthreads*chunksize, agent.size(), ref(agent), ref(*game), ref(evaluations)));
			}
			for (thread& t : threads) t.join(); // wait for all threads to finish
		}

		/// perform selection
		maxFitness = 0.0f;
		for(i=0;i<agent.size();i++) {
			tempFitness = 1.0f;
			if ((selectionRegime & (int)Selection::task) == (int)Selection::task) {
            switch(gameID) {
               case 0:
                  tempFitness *= pow(fitnessPower,agent[i]->correct - agent[i]->incorrect);
                  break;
               case 1:
                  tempFitness *= pow(fitnessPower,(agent[i]->fitness / 120.0f) * 80.0f);
                  break;
            }
			}
			if ((selectionRegime & (int)Selection::phi) == (int)Selection::phi) {
            switch(gameID) {
               case 0:
               case 1:
                  tempFitness *= agent[i]->Phi/16.0f + 1.0f;
                  break;
            }
			}
			if ((selectionRegime & (int)Selection::r) == (int)Selection::r) {
            switch(gameID) {
               case 0:
                  tempFitness *= agent[i]->R/4.0f + 1.0f;
                  break;
               case 1:
                  // Do nothing for R
                  break;
            }
			}
			if ((selectionRegime & (int)Selection::diameter) == (int)Selection::diameter) {
            switch(gameID) {
               case 0:
               case 1: // do same for both
                  tempFitness *= (agent[i]->Diameter / 16.0f) + 1.0f;
                  break;
            }
			}
         if ((selectionRegime & (int)Selection::connectedness) == (int)Selection::connectedness) {
            switch(gameID) {
               case 0:
               case 1: // do same for both
                  tempFitness *= agent[i]->Connectedness + 1.0f;
                  break;
            }
         }
         if ((selectionRegime & (int)Selection::sparseness) == (int)Selection::sparseness) {
            switch(gameID) {
               case 0:
               case 1: // do same for both
                  tempFitness *= agent[i]->Sparseness + 1.0f;
                  break;
            }
         }
			if ((selectionRegime & (int)Selection::genome) == (int)Selection::genome) {
            switch(gameID) {
               case 0:
               case 1: // do same for both
                  tempFitness *= (agent[i]->genome.size() / 20000.0f) + 1.0f;
                  break;
            }
			}
         if ((selectionRegime & (int)Selection::predIs2s) == (int)Selection::predIs2s) {
            switch(gameID) {
               case 0:
               case 1:
                  tempFitness *= agent[i]->PIs2s + 1.0f;
                  break;
            }
         }
         if ((selectionRegime & (int)Selection::predIs2m) == (int)Selection::predIs2m) {
            switch(gameID) {
               case 0:
               case 1:
                  tempFitness *= agent[i]->PIs2m + 1.0f;
                  break;
            }
         }
			if ((selectionRegime & (int)Selection::none) == (int)Selection::none) {
				tempFitness = 2.0f;
				agent[i]->correct = 0;
				agent[i]->incorrect = 0;
				agent[i]->fitness = 1.0f;
				agent[i]->R = 1.0f;
				agent[i]->Phi = 1.0f;
            agent[i]->Connectedness = 1.0f;
            agent[i]->Sparseness = 1.0f;
			}
			agent[i]->fitness = tempFitness;
			if (tempFitness > maxFitness) {
				maxFitness = tempFitness;
				who = i;
			}
		}
		if ((selectionRegime & (int)Selection::none) == (int)Selection::none) {
			maxFitness = -1.0f;
		}

		/// regime switch conditions
		if ((nregimes > 1) || (selectionRegime != (int)Selection::task)) {
			if ((regimeGenLimit != -1) && (update > regimeGenLimit)) {
				selectionRegime = (int)Selection::task;
				nregimes = 1;
			}
			if (regimeValueLimit > 0.0f) {
				if (maxFitness > pow(fitnessPower, regimeValueLimit*80.0f)) {
					selectionRegime = (int)Selection::task;
					nregimes = 1;
					if (stopOnLimit) break;
				}
			}
		}
			// convert maxValue to 0-100
  printf("%i %f %f %f %f %f %f %f %f %i %i\n", update, maxFitness, agent[who]->Phi, agent[who]->R, agent[who]->Diameter, agent[who]->Connectedness, agent[who]->Sparseness, agent[who]->PIs2s, agent[who]->PIs2m, agent[who]->correct, agent[who]->incorrect);

		int j=0;
		for(i=0;i<agent.size();i++) {
			tAgent *d;
			d=new tAgent;
			if(maxFitness<=0.0f){
				j=rand()%(int)agent.size();
			} else {
				do{
					j=rand()%(int)agent.size();
				} while((j==(int)i)||(randDouble>( agent[j]->fitness / maxFitness )));
			}
			d->inherit(agent[j],perSiteMutationRate,update);
			nextGen[i]=d;
		}
	//}
		for(i=0;i<agent.size();i++){
			agent[i]->retire();
			agent[i]->nrPointingAtMe--;
			if(agent[i]->nrPointingAtMe==0)
				delete agent[i];
			agent[i]=nextGen[i];
		}
		agent=nextGen;
		update++;
	}
	
	agent[0]->ancestor->saveLOD(LODFile,cstr(filenameGenome), experimentID, replicateID, -1); // -1 to tell saveLOD to make header for csv
	if (stopOnLimit) {
		float maxPhi=0.0;
		tAgent* bestAgent=nullptr;
		for (tAgent* a : agent) {
			if (a->Phi > maxPhi) {
				maxPhi = a->Phi;
				bestAgent = a;
			}
		}
		if (bestAgent) {
			bestAgent->saveGenome(genomeFile);
		}
	} else {
		agent[0]->ancestor->ancestor->saveGenome(genomeFile);
	}
//	agent[0]->ancestor->saveToDot(argv[3]);
	agent.clear();
	nextGen.clear();
	delete masterAgent;
	delete game;
	return 0;
}

void computeLOD(FILE *f,FILE *g, tAgent *agent,tGame *game){
	/*vector<vector<int> > table;
	double R,oldR;
	if(agent->ancestor!=NULL)
		computeLOD(f,g,agent->ancestor,game);
	agent->setupPhenotype();
	table=game->executeGame(agent, 2, NULL,false,-1,-1);
	R=game->computeR(table,0);
	oldR=game->computeOldR(table);
	fprintf(f,"%i	%i	%i	%f	%f",agent->ID,agent->correct,agent->incorrect,agent->extra);
	fprintf(f,"\n");
	*/
}

