/*
 *  tAgent.h
 *  HMMBrain
 *
 *  Modified from code by Arend Hintze Copyright 2010
 *
 */

#pragma once

#ifndef _tAgent_h_included_
#define _tAgent_h_included_

#include "globalConst.h"
#include "tHMM.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>


using namespace std;

static int masterID=0;

class tDot{
public:
	double xPos,yPos;
};


class tAgent{
public:
	vector<tHMMU*> hmmus;
	vector<unsigned char> genome;
	vector<tDot> dots;
	double extra;
	double H,nH,Phi,R,Diameter,Connectedness,Sparseness,PIs2s,PIs2m;
	tAgent *ancestor;
	unsigned int nrPointingAtMe;
	unsigned char states[maxNodes],newStates[maxNodes];
	double fitness,convFitness,phitness;
	vector<double> fitnesses;
	int food;
	
	double xPos,yPos,direction;
	double sX,sY;
	bool foodAlreadyFound;
	int steps,bestSteps,totalSteps;
	int ID,nrOfOffspring;
	bool saved;
	bool retired;
	int born;
	int correct,incorrect;
    
	tAgent();
	~tAgent();
	void setupRandomAgent(int nucleotides);
	void loadAgent(const char* filename);
	void loadAgentWithTrailer(const char* filename);
	void setupPhenotype(void);
	void inherit(tAgent *from,double mutationRate,int theTime);
	unsigned char * getStatesPointer(void);
	void updateStates(void);
	void resetBrain(void);
	void ampUpStartCodons(void);
	void showBrain(void);
	void showPhenotype(void);
	void saveToDot(const char *filename);
	void saveToDotFullLayout(const char *filename);
   void saveEdgeList(const char *filename);
   double gammaIndex();
	vector<vector<int> > getBrainMap(void);
	vector<vector<int> > getDistMap(vector<vector<int> > M);
	
	void initialize(int x, int y, int d);
	tAgent* findLMRCA(void);
	void saveFromLMRCAtoNULL(FILE *statsFile,FILE *genomeFile);
	void saveLOD_recursive(FILE *statsFile, const char* genomeFileNameBase, string experimentID, int replicateID, int progenitorDOB, int& iter_count);
	void saveLOD(FILE *statsFile, const char* genomeFile, string experimentID, int replicateID, int progenitorDOB);
	void retire(void);
	void setupDots(int x, int y,double spacing);
	void saveLogicTable(FILE *f);
	void saveGenome(FILE *f);
};

#endif
