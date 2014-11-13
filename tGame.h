/*
 *  tGame.h
 *  HMMBrain
 *
 *  Modified from code by Arend Hintze Copyright 2010
 *
 */

#pragma once
 
#ifndef _tGame_h_included_
#define _tGame_h_included_

#include "globalConst.h"
#include "tAgent.h"
#include <vector>
#include <map>
#include <set>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define xDim 256
#define yDim 16
#define startMazes 1
#define cPI 3.14159265

class tGame{
	public:
		vector<vector<int> > executeGame(tAgent* agent,int paddleWidth,FILE *f,bool logStates,int ko, int setTo);
		tGame();
		~tGame();
		void getTableForPhi(tAgent *agent,char *filename);
		void represenationDecomposition(tAgent* agent,int paddleWidth,char* filename);
		void represenationPerNodeSummary(tAgent* agent,int paddleWidth,char* filename);
		void makeFullAnalysis(tAgent *agent,char *fileLead);
		double mutualInformation(vector<int> A,vector<int>B);
		double ei(vector<int> A,vector<int> B,int theMask);
		double computeAtomicPhi(vector<int> A,int states);
		double predictiveI(vector<int>A);
		double nonPredictiveI(vector<int>A);
		double predictNextInput(vector<int>A);
		double computeR(vector<vector<int> > table,size_t howFarBack);
		double computeOldR(vector<vector<int> > table);
		double entropy(vector<int> list);
		double computeRGiven(vector<int>W,vector<int>S,vector<int>B,int nrWstates,int nrSstates,int nrBstates);
		double brainEntropy(vector<int> states);

};
#endif
