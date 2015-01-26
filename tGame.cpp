/*
 *  tGame.cpp
 *  HMMBrain
 *
 *  Modified from code by Arend Hintze Copyright 2010
 *
 */

#include "tGame.h"
#include <math.h>
#include <numeric>
#include "globalConst.h"



tGame::tGame(){
}

tGame::~tGame(){
}


vector<vector<int> > tGame::executeGame(tAgent* agent,int paddleWidth,FILE *f,bool logStates,int ko, int setTo){
	int spB,w,d,u;
	int Ox,Ax;
	int grid[20];
	int i;
	bool hit;
	int correct,incorrect;
	int E,I,H,M,T0,T1;
	vector<vector<int> > retValue,RValue;
	vector<int> dummyE,dummyI,dummyH,dummyM,dummyT0,dummyT1,R0,R1;
	correct=0;
	incorrect=0;
	agent->fitness=1.0;
	agent->convFitness=0.0;
	for(spB=0;spB<20;spB++){
		for(w=2;w<6;w+=2){
			for(d=-1;d<3;d+=2){
				Ax=0;
				Ox=spB;
				agent->resetBrain();
				for(u=0;u<20;u++){
					hit=false;
					for(i=0;i<20;i++)
						grid[i]=0;
					for(i=0;i<w;i++)
						grid[(Ox+i+20)%20]=1;
					for(i=0;i<4+paddleWidth;i++)
						if(grid[(Ax+i)%20]==1) 
							hit=true;
					agent->states[0]=grid[Ax];
					agent->states[1]=grid[(Ax+1)%20];
					agent->states[2]=grid[(Ax+paddleWidth+2)%20];
					agent->states[3]=grid[(Ax+paddleWidth+3)%20];
					agent->states[maxNodes-1]=0;
					agent->states[maxNodes-2]=0;
					if(ko!=-1)
						agent->states[ko]=setTo;
					E=0;
					I=0;
					H=0;
					M=0;
					if(w==4) E+=8;
					if(d==1) E+=4;
					int cm=(Ox+(w/2))-(Ax+4+paddleWidth);
					if(cm>=0) E+=2;
					if(hit) E+=1;
					for(i=0;i<4;i++)
						I=(I<<1)+agent->states[i];
					for(i=4;i<14;i++)
						H=(H<<1)+agent->states[i];
					for(i=14;i<16;i++)
						M=(M<<1)+agent->states[i];
					dummyE.push_back(E);
					dummyI.push_back(I);

					T0=0;
					for(i=0;i<16;i++)
						T0|=(agent->states[i]&1)<<i;

					agent->updateStates();

					T1=0;
					for(i=0;i<16;i++)
						T1|=(agent->states[i]&1)<<i;

					H=0;
					M=0;
					for(i=4;i<16;i++)
						H=(H<<1)+agent->states[i];
					for(i=14;i<16;i++)
						M=(M<<1)+agent->states[i];
					dummyH.push_back(H);
					dummyM.push_back(M);
					dummyT0.push_back((E<<16)+(I<<12)+H);
					R0.push_back(T0);
					R1.push_back(T1);

					int action=(agent->states[maxNodes-1])+(agent->states[maxNodes-2]<<1);
					switch (action) {
						case 0: case 3: break;
						case 1:Ax=(Ax+1)%20;break;
						case 2:Ax=(Ax-1+20)%20;break;
					}
					Ox=(Ox+20+d)%20;
					if(f!=NULL)
						fprintf(f,"%i,%i,%i,%i\n",Ox,u,w,Ax);

				}
				if(w==2){
					if(hit){
						agent->fitness*=fitnessPower;
						agent->convFitness+=1.0;
						correct++;
					}
					else {
						agent->convFitness-=1.0;
						agent->fitness/=fitnessPower;
						incorrect++;
					}

				}
				else{
					if(!hit){
						agent->fitness*=fitnessPower;
						agent->convFitness+=1.0;
						correct++;
					}
					else {
						agent->convFitness-=1.0;
						agent->fitness/=fitnessPower;
						incorrect++;
					}
				}
			}
		}
	}
	agent->correct=correct;
	agent->incorrect=incorrect;
	if(logStates){
		retValue.resize(2);
		retValue[0]=R0;
		retValue[1]=R1;
	} else {
		retValue.push_back(dummyE);
		retValue.push_back(dummyI);
		retValue.push_back(dummyH);
		retValue.push_back(dummyM);
		retValue.push_back(dummyT0);
	}
	RValue.push_back(dummyE); // env
	RValue.push_back(dummyI); // internal
	RValue.push_back(dummyH); //hiddne
	RValue.push_back(dummyM); // 
	RValue.push_back(dummyT0); // brain state

	agent->Phi=computeAtomicPhi(retValue[0], maxNodes);
	agent->R=computeR(RValue,0);
	agent->extra=1.0;
	vector<vector<int>> paths = agent->getDistMap(agent->getBrainMap());
	int longestPath(0);
	int sumPath(0);
	for (auto& row : paths) {
		sumPath=accumulate(begin(row), end(row), 0);
		if (sumPath > longestPath) longestPath = sumPath;
	}
	agent->Topology = longestPath;
	return retValue;
}

double tGame::mutualInformation(vector<int> A,vector<int>B){
	set<int> nrA,nrB;
	set<int>::iterator aI,bI;
	map<int,map<int,double> > pXY;
	map<int,double> pX,pY;
	size_t i;
	double c=1.0/(double)A.size();
	double I=0.0;
	for(i=0;i<A.size();i++){
		nrA.insert(A[i]);
		nrB.insert(B[i]);
		pX[A[i]]=0.0;
		pY[B[i]]=0.0;
	}
	for(aI=nrA.begin();aI!=nrA.end();aI++)
		for(bI=nrB.begin();bI!=nrB.end();bI++){
			pXY[*aI][*bI]=0.0;
		}
	for(i=0;i<A.size();i++){
		pXY[A[i]][B[i]]+=c;
		pX[A[i]]+=c;
		pY[B[i]]+=c;
	}
	for(aI=nrA.begin();aI!=nrA.end();aI++)
		for(bI=nrB.begin();bI!=nrB.end();bI++)
			if((pX[*aI]!=0.0)&&(pY[*bI]!=0.0)&&(pXY[*aI][*bI]!=0.0))
				I+=pXY[*aI][*bI]*log2(pXY[*aI][*bI]/(pX[*aI]*pY[*bI]));
	return I;

}

double tGame::entropy(vector<int> list){
	map<int, double> p;
	map<int,double>::iterator pI;
	size_t i;
	double H=0.0;
	double c=1.0/(double)list.size();
	for(i=0;i<list.size();i++)
		p[list[i]]+=c;
	for (pI=p.begin();pI!=p.end();pI++) {
		H+=p[pI->first]*log2(p[pI->first]);	
	}
	return -1.0*H;
}

double tGame::ei(vector<int> A,vector<int> B,int theMask){
	set<int> nrA,nrB;
	set<int>::iterator aI,bI;
	map<int,map<int,double> > pXY;
	map<int,double> pX,pY;
	size_t i;
	double c=1.0/(double)A.size();
	double I=0.0;
	for(i=0;i<A.size();i++){
		nrA.insert(A[i]&theMask);
		nrB.insert(B[i]&theMask);
		pX[A[i]&theMask]=0.0;
		pY[B[i]&theMask]=0.0;
	}
	for(aI=nrA.begin();aI!=nrA.end();aI++)
		for(bI=nrB.begin();bI!=nrB.end();bI++){
			pXY[*aI][*bI]=0.0;
		}
	for(i=0;i<A.size();i++){
		pXY[A[i]&theMask][B[i]&theMask]+=c;
		pX[A[i]&theMask]+=c;
		pY[B[i]&theMask]+=c;
	}
	for(aI=nrA.begin();aI!=nrA.end();aI++)
		for(bI=nrB.begin();bI!=nrB.end();bI++)
			if((pX[*aI]!=0.0)&&(pY[*bI]!=0.0)&&(pXY[*aI][*bI]!=0.0))
				I+=pXY[*aI][*bI]*log2(pXY[*aI][*bI]/(pY[*bI]));
	return -I;
}
double tGame::computeAtomicPhi(vector<int>A, int states){
	int i;
	double P, EIsystem;
	vector<int> T0, T1;
	T0 = A;
	T1 = A;
	T0.erase(T0.begin() + T0.size() - 1);
	T1.erase(T1.begin());
	EIsystem = ei(T0, T1, (1 << states) - 1);
	P = 0.0;
	for (i = 0; i<states; i++){
		double EIP = ei(T0, T1, 1 << i);
		//		cout<<EIP<<endl;
		P += EIP;
	}
	//	cout<<-EIsystem+P<<" "<<EIsystem<<" "<<P<<" "<<T0.size()<<" "<<T1.size()<<endl;
	return -EIsystem + P;
}

/*
	double tGame::computeAtomicPhi(vector<int> T0,vector<int> T1,int states){
	int i;
	double P,EIsystem;
	EIsystem=ei(T0,T1,(1<<states)-1);
	P=0.0;
	for(i=0;i<states;i++){
	double EIP=ei(T0,T1,1<<i);
//		cout<<EIP<<endl;
P+=EIP;
}
//	cout<<-EIsystem+P<<" "<<EIsystem<<" "<<P<<" "<<T0.size()<<" "<<T1.size()<<endl;
return -EIsystem+P;
}
*/


double tGame::computeR(vector<vector<int> > table,size_t howFarBack){
	double Iwh,Iws,Ish,Hh,Hs,Hw,Hhws,delta,R;
	size_t i;
	for(i=0;i<howFarBack;i++){
		table[0].erase(table[0].begin());
		table[1].erase(table[1].begin());
		table[2].erase(table[2].begin()+(table[2].size()-1));
	}
	table[4].clear();
	for(i=0;i<table[0].size();i++){
		table[4].push_back((table[0][i]<<14)+(table[1][i]<<10)+table[2][i]);
	}
	Iwh=mutualInformation(table[0],table[2]);
	Iws=mutualInformation(table[0],table[1]);
	Ish=mutualInformation(table[1],table[2]);
	Hh=entropy(table[2]);
	Hs=entropy(table[1]);
	Hw=entropy(table[0]);
	Hhws=entropy(table[4]);
	delta=Hhws+Iwh+Iws+Ish-Hh-Hs-Hw;
	R=Iwh-delta;
	return R;
}

double tGame::computeRGiven(vector<int>W,vector<int>S,vector<int>B,int nrWstates,int nrSstates,int nrBstates){
	double Iwh,Iws,Ish,Hh,Hs,Hw,Hhws,delta,R;
	size_t i;
	vector<int> total;
	total.clear();
	for(i=0;i<W.size();i++){
		total.push_back((W[i]<<(nrBstates+nrWstates))+(S[i]<<nrBstates)+B[i]);
	}
	Iwh=mutualInformation(W,B);
	Iws=mutualInformation(W,S);
	Ish=mutualInformation(S,B);
	Hh=entropy(B);
	Hs=entropy(S);
	Hw=entropy(W);
	Hhws=entropy(total);
	delta=Hhws+Iwh+Iws+Ish-Hh-Hs-Hw;
	R=Iwh-delta;
	return R;

}

double tGame::computeOldR(vector<vector<int> > table){
	double Ia,Ib;
	Ia=mutualInformation(table[0], table[2]);
	Ib=mutualInformation(table[1], table[2]);
	return Ib-Ia;
}

double tGame::predictiveI(vector<int>A){
	vector<int> S,I;
	S.clear(); I.clear();
	for(size_t i=0;i<A.size();i++){
		S.push_back((A[i]>>12)&15);
		I.push_back(A[i]&3);
	}
	return mutualInformation(S, I);
}

double tGame::nonPredictiveI(vector<int>A){
	vector<int> S,I;
	S.clear(); I.clear();
	for(size_t i=0;i<A.size();i++){
		S.push_back((A[i]>>12)&15);
		I.push_back(A[i]&3);
	}
	return entropy(I)-mutualInformation(S, I);
}
double tGame::predictNextInput(vector<int>A){
	vector<int> S,I;
	S.clear(); I.clear();
	for(size_t i=0;i<A.size();i++){
		S.push_back((A[i]>>12)&15);
		I.push_back(A[i]&3);
	}
	S.erase(S.begin());
	I.erase(I.begin()+I.size()-1);
	return mutualInformation(S, I);
}

void tGame::represenationDecomposition(tAgent* const agent,int paddleWidth,char* filename){
	vector<vector<int> > table=executeGame(agent, paddleWidth, NULL,false,-1,-1);
	int W,B;
	size_t i,j;
	double R;
	int bitsumW,bitsumB;
	FILE *F=fopen(filename,"w+t");
	vector<int> world,sensors,brain;
	cout<<"fitness of agent: "<<agent->correct<<endl;
	world.resize(table[0].size());
	sensors.resize(table[0].size());
	brain.resize(table[0].size());
	for(W=1;W<16;W++){
		for(B=1;B<1024;B++){
			cout<<W<<" "<<B<<" ";
			fprintf(F,"%i   %i  ",W,B);
			bitsumW=0;
			bitsumB=0;
			for(i=0;i<4;i++){
				cout<<((W>>i)&1)<<" ";
				fprintf(F,"%i   ",((W>>i)&1));
				bitsumW+=(W>>i)&1;
			}
			for(i=0;i<10;i++){
				cout<<((B>>i)&1)<<" ";
				fprintf(F,"%i   ",((B>>i)&1));
				bitsumB+=(B>>i)&1;
			}
			cout<<bitsumW<<" "<<bitsumB<<" ";
			fprintf(F,"%i   %i  ",bitsumW,bitsumB);
			for(j=0;j<table[0].size();j++){
				world[j]=table[0][j]&W;
				sensors[j]=table[1][j];
				brain[j]=table[2][j]&B;
			}
			R=computeRGiven(world, sensors, brain, 4,4,10);
			if(R<0.0) R=0.0;
			cout<<R<<endl;
			fprintf(F,"%f\n",R);
		}
	}
	fclose(F);
}

void tGame::represenationPerNodeSummary(tAgent* agent,int paddleWidth,char* filename){
	vector<vector<int> > table=executeGame(agent, paddleWidth, NULL,false,-1,-1);
	int W,B;
	size_t i,j;
	double R;
	int bitsumW,bitsumB;
	double maxR;
	int minP,bestPartition;
	FILE *F=fopen(filename,"w+t");
	vector<int> world,sensors,brain;
	cout<<"fitness of agent: "<<agent->correct<<endl;
	world.resize(table[0].size());
	sensors.resize(table[0].size());
	brain.resize(table[0].size());
	//    for(W=1;W<16;W=W<<1){
	for(W=1;W<16;W++){
		maxR=0.0;
		bestPartition=0;
		for(B=1;B<1024;B++){
			//cout<<W<<" "<<B<<" ";
			//fprintf(F,"%i   %i  ",W,B);
			bitsumW=0;
			bitsumB=0;
			for(i=0;i<4;i++){
				bitsumW+=(W>>i)&1;
			}
			for(i=0;i<10;i++){
				bitsumB+=(B>>i)&1;
			}
			for(j=0;j<table[0].size();j++){
				world[j]=table[0][j]&W;
				sensors[j]=table[1][j];
				brain[j]=table[2][j]&B;
			}
			R=computeRGiven(world, sensors, brain, 4,4,10);
			if(R<0.0) R=0.0;
			if(R==maxR){
				if(bitsumB<minP){
					minP=bitsumB;
					bestPartition=B;
				}
			}
			if(R>maxR){
				maxR=R;
				minP=bitsumB;
				bestPartition=B;
			}
		}
		cout<<W<<" "<<bestPartition<<" ";
		fprintf(F,"%i   %i  ",W,bestPartition);
		for(i=0;i<4;i++){
			cout<<((W>>i)&1)<<" ";
			fprintf(F,"%i   ",((W>>i)&1));
		}
		for(i=0;i<10;i++){
			cout<<((bestPartition>>i)&1)<<" ";
			fprintf(F,"%i   ",((bestPartition>>i)&1));
		}
		cout<<R<<endl;
		fprintf(F,"%f\n",R);
	}
	fclose(F);
}

void tGame::getTableForPhi(tAgent *agent,char *filename){
	FILE *f;
	size_t j;
	vector<vector<int> > table;
	//state to state table for only the lifetime
	f=fopen(filename,"w+t");
	table=this->executeGame(agent, 2,NULL,true,-1,-1);
	fprintf(f,"16\n");
	for(j=0;j<table[0].size();j++){
		fprintf(f,"%i   %i\n",table[0][j],table[1][j]);
	}    
	fclose(f);
}
void tGame::makeFullAnalysis(tAgent *agent,char *fileLead){
	char filename[1000];
	FILE *f;
	size_t i,j;
	vector<vector<int> > table;

	//representation table
	sprintf(filename,"%s_representation.txt",fileLead);
	represenationPerNodeSummary(agent, 2, filename);
	//state to state table
	sprintf(filename,"%s_FullLogicTable.txt",fileLead);
	f=fopen(filename,"w+t");
	agent->saveLogicTable(f);
	fclose(f);

	//state to state table for only the lifetime
	sprintf(filename,"%s_LifetimeLogicTable.txt",fileLead);
	f=fopen(filename,"w+t");
	table=this->executeGame(agent, 2,NULL,true,-1,-1);
	for(i=0;i<16;i++)
		fprintf(f,"T0_%i,",(int)i);
	fprintf(f,",");
	for(i=0;i<16;i++)
		fprintf(f,"T1_%i,",(int)i);
	fprintf(f,"\n");
	for(j=0;j<table[0].size();j++){
		for(i=0;i<16;i++)
			fprintf(f,"%i,",(table[0][j]>>i)&1);
		fprintf(f,",");
		for(i=0;i<16;i++)
			fprintf(f,"%i,",(table[1][j]>>i)&1);
		fprintf(f,"\n");
	}
	fclose(f);

	//ko table
	sprintf(filename,"%s_KOdata.txt",fileLead);
	f=fopen(filename,"w+t");
	executeGame(agent, 2,NULL,true,-1,-1);
	fprintf(f,"%i",agent->correct);
	for(i=0;i<16;i++)
		for(j=0;j<2;j++){
			executeGame(agent, 2,NULL,false,i,j);
			fprintf(f,"	%i",agent->correct);
		}
	fprintf(f,"\n");
	fclose(f);
	//dot file
	sprintf(filename,"%s_EdgeList.txt",fileLead);
	agent->saveEdgeList(filename);

}


double tGame::brainEntropy(vector<int> states){
	double H=0.0;
	vector<int> L;
	size_t i,j;
	for(j=0;j<maxNodes;j++){
		L.clear();
		for(i=0;i<states.size();i++)
			L.push_back((states[i]>>j)&1);
		H+=entropy(L);
	}
	return H;
}



