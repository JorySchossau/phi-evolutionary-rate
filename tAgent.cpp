/*
 *  tAgent.cpp
 *  HMMBrain
 *
 *  Modified from code by Arend Hintze Copyright 2010
 *
 */

#include "tAgent.h"

tAgent::tAgent(){
	int i;
	nrPointingAtMe=1;
	ancestor=NULL;
	for(i=0;i<maxNodes;i++){
		states[i]=0;
		newStates[i]=0;
	}
	bestSteps=-1;
	ID=masterID;
	masterID++;
	saved=false;
	hmmus.clear();
	nrOfOffspring=0;
	totalSteps=0;
	retired=false;
	Diameter=0;
	food=0;
   PIs2m=0.0;
   PIs2s=0.0;
}

tAgent::~tAgent(){
	for(size_t i=0;i<hmmus.size();i++)
		delete hmmus[i];
	if(ancestor!=NULL){
		ancestor->nrPointingAtMe--;
		if(ancestor->nrPointingAtMe==0)
			delete ancestor;
	}
}

void tAgent::setupRandomAgent(int nucleotides){
	int i;
	genome.resize(nucleotides);
	for(i=0;i<nucleotides;i++)
		genome[i]=127;//rand()&255;
	ampUpStartCodons();
	setupPhenotype();
}
void tAgent::loadAgent(const char* filename){
	FILE *f=fopen(filename,"r+t");
	int i;
	genome.clear();
	while(!(feof(f))){
		fscanf(f,"%i	",&i);
		genome.push_back((unsigned char)(i&255));
	}
	setupPhenotype();
}
void tAgent::loadAgentWithTrailer(const char* filename){
	FILE *f=fopen(filename,"r+t");
	int i;
	genome.clear();
	fscanf(f,"%i	",&i);
	while(!(feof(f))){
		fscanf(f,"%i	",&i);
		genome.push_back((unsigned char)(i&255));
	}
	setupPhenotype();
}


void tAgent::ampUpStartCodons(void){
	size_t i,j;
	for(i=0;i<genome.size();i++)
		genome[i]=rand()&255;
	for(i=0;i<4;i++)
	{
		j=rand()%(genome.size()-100);
		genome[j]=42;
		genome[j+1]=(255-42);
		for(int k=2;k<20;k++)
			genome[j+k]=rand()&255;
	}
}

void tAgent::inherit(tAgent *from,double mutationRate,int theTime){ // large mutation rate to spread the genotypic population
	int nucleotides=from->genome.size();
	int i,s,o,w;
	//double localMutationRate=4.0/from->genome.size();
	vector<unsigned char> buffer;
	born=theTime;
	ancestor=from;
	from->nrPointingAtMe++;
	from->nrOfOffspring++;
	genome.clear();
	genome.resize(from->genome.size());
	for(i=0;i<nucleotides;i++)
		if(((double)rand()/(double)RAND_MAX)<mutationRate)
			genome[i]=rand()&255;
		else
			genome[i]=from->genome[i];
	if((((double)rand()/(double)RAND_MAX)<0.05)&&(genome.size()<20000)){
		//duplication
		w=15+rand()&511;
		s=rand()%(genome.size()-w);
		o=rand()%genome.size();
		buffer.clear();
		buffer.insert(buffer.begin(),genome.begin()+s,genome.begin()+s+w);
		genome.insert(genome.begin()+o,buffer.begin(),buffer.end());
	}
	if((((double)rand()/(double)RAND_MAX)<0.05)&&(genome.size()>1000)){
	//if((((double)rand()/(double)RAND_MAX)<0.02)&&(genome.size()>1000)){
		//deletion
		w=15+rand()&511;
		s=rand()%(genome.size()-w);
		genome.erase(genome.begin()+s,genome.begin()+s+w);
	}
	setupPhenotype();
	fitness=0.0;
	phitness=0.0;
	Phi=0.0;
	Diameter=0.0;
   Connectedness=0.0;
   Sparseness=0.0;
   PIs2m=0.0;
   PIs2s=0.0;
}
void tAgent::setupPhenotype(void){
	size_t i;
	tHMMU *hmmu;
	if(hmmus.size()!=0)
		for(i=0;i<hmmus.size();i++)
			delete hmmus[i];
	hmmus.clear();
	for(i=0;i<genome.size();i++){
		if((genome[i]==42)&&(genome[(i+1)%genome.size()]==(255-42))){
			hmmu=new tHMMU;
			hmmu->setupQuick(genome,i); // deterministic organism
			hmmus.push_back(hmmu);
		}
	}
}

void tAgent::retire(void){
	if((born&255)==0)
		retired=false;
	else
		retired=true;
}

unsigned char * tAgent::getStatesPointer(void){
	return states;
}

void tAgent::resetBrain(void){
	for(int i=0;i<maxNodes;i++)
		states[i]=0;
}

void tAgent::updateStates(void){
	size_t i;
	for(i=0;i<hmmus.size();i++)
		hmmus[i]->update(&states[0],&newStates[0]);
	for(i=0;i<maxNodes;i++){
		states[i]=newStates[i];
		newStates[i]=0;
	}
	totalSteps++;
}

void tAgent::showBrain(void){
	for(int i=0;i<maxNodes;i++)
		cout<<(int)states[i];
	cout<<endl;
}

void tAgent::initialize(int x, int y, int d){
	xPos=x;
	yPos=y;
	direction=d;
	steps=0;
}

tAgent* tAgent::findLMRCA(void){
	tAgent *r,*d;
	if(ancestor==NULL)
		return NULL;
	else{
		r=ancestor;
		d=NULL;
		while(r->ancestor!=NULL){
			if(r->ancestor->nrPointingAtMe!=1)
				d=r;
			r=r->ancestor;
		}
		return d;
	}
}

void tAgent::saveFromLMRCAtoNULL(FILE *statsFile,FILE *genomeFile){
	//if(ancestor!=NULL)
	//	ancestor->saveFromLMRCAtoNULL(statsFile,genomeFile);
	//if(!saved){ 
	//	saveLOD(statsFile,"%i	%i	%i	%f	%i	%f	%i	%i\n",ID,born,(int)genome.size(),fitness,bestSteps,(float)totalSteps/(float)nrOfOffspring,correct,incorrect);
	//	fprintf(genomeFile,"%i	",ID);
	//	for(size_t i=0;i<genome.size();i++)
	//		fprintf(genomeFile,"	%i",genome[i]);
	//	fprintf(genomeFile,"\n");
	//	saved=true;
	//}
	//if((saved)&&(retired)) genome.clear();
}
void tAgent::saveLOD_recursive(FILE *statsFile, const char* genomeFileNameBase, string experimentID, int replicateID, int progenitorDOB, int& iter_count){
	if (progenitorDOB==-1) { // we always pass -1 when invoking from main
  fprintf(statsFile,"%s %s %s %s %s %s %s %s %s %s %s %s %s %s\n","DOB","genome_size","fitness","correct","incorrect","phi","r","diameter","connectedness","sparseness","pis2s","pis2m","experimentID","replicateID");
	}
	//if(iter_count&255 == 255){
   //   FILE *genomeFile = fopen((genomeFileNameBase+string(".")+std::to_string(iter_count)).c_str(), "w+t");
	//	for(int i=0;i<genome.size();i++)
	//		fprintf(genomeFile,"	%i",genome[i]);
	//	fprintf(genomeFile,"\n");
   //   fclose(genomeFile);
	//}
	if(ancestor!=NULL) {
		ancestor->saveLOD_recursive(statsFile,genomeFileNameBase, experimentID, replicateID, born, ++iter_count);
	} 
	for (int i=born; i<progenitorDOB; ++i) {
  fprintf(statsFile,"%i %i %f %i %i %f %f %f %f %f %f %f %s %i\n",born,(int)genome.size(),fitness,correct,incorrect,Phi,R,Diameter,Connectedness,Sparseness,PIs2s,PIs2m,experimentID.c_str(),replicateID);
	}
	
}

void tAgent::saveLOD(FILE *statsFile, const char* genomeFileNameBase, string experimentID, int replicateID, int progenitorDOB){
   int iter_count = 0;
   saveLOD_recursive(statsFile, genomeFileNameBase, experimentID, replicateID, progenitorDOB, iter_count);
}

void tAgent::showPhenotype(void){
	for(size_t i=0;i<hmmus.size();i++)
		hmmus[i]->show();
	cout<<"------"<<endl;
}

void tAgent::saveToDot(const char *filename){
	FILE *f=fopen(filename,"w+t");
	size_t i,j,k;
	fprintf(f,"digraph brain {\n");
	fprintf(f,"	ranksep=2.0;\n");
	for(i=0;i<4;i++)
		fprintf(f,"	%i [shape=invtriangle,style=filled,color=red];\n",(int)i);
	for(i=4;i<14;i++)
		fprintf(f,"	%i [shape=circle,color=blue];\n",(int)i);
	for(i=14;i<16;i++)
		fprintf(f,"	%i [shape=circle,style=filled,color=green];\n",(int)i);
	for(i=0;i<hmmus.size();i++){
	//	fprintf(f,"	{\n");
		for(j=0;j<hmmus[i]->ins.size();j++){
			for(k=0;k<hmmus[i]->outs.size();k++)
				fprintf(f,"	%i	->	%i;\n",hmmus[i]->ins[j],hmmus[i]->outs[k]);
		}
	//	fprintf(f,"	}\n");
	}
	fprintf(f,"	{ rank=same; 0; 1; 2; 3;}\n"); 
	fprintf(f,"	{ rank=same; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13;}\n"); 
	fprintf(f,"	{ rank=same; 14; 15; }\n"); 
	fprintf(f,"}\n");
	fclose(f);
}

void tAgent::saveEdgeList(const char *filename){
	FILE *f=fopen(filename,"w+t");
	size_t i,j,k;
	for(i=0;i<hmmus.size();i++){
		for(j=0;j<hmmus[i]->ins.size();j++){
			for(k=0;k<hmmus[i]->outs.size();k++)
				fprintf(f,"%i   %i\n",hmmus[i]->ins[j],hmmus[i]->outs[k]);
		}
	}
	fclose(f);
}

double tAgent::gammaIndex() { // connectivity of brain
    int M[maxNodes][maxNodes];
    int i,j,k;
    for(i=0;i<maxNodes;i++)
        for(j=0;j<maxNodes;j++)
            M[i][j]=0;
    for(i=0;i<(int)hmmus.size();i++)
        for(j=0;j<(int)hmmus[i]->ins.size();j++)
            for(k=0;k<(int)hmmus[i]->outs.size();k++)
                M[hmmus[i]->ins[j]][hmmus[i]->outs[k]]=1;
    k=0;
    for(i=0;i<maxNodes;i++)
        for(j=0;j<maxNodes;j++)
            k+=M[i][j];
    return (double)k/(double)(maxNodes*maxNodes);
}

vector<vector<int> > tAgent::getBrainMap(void){
    vector<vector<int> > M;
    int i,j,k;
    M.clear();
    M.resize(maxNodes);
    for(i=0;i<maxNodes;i++){
        M[i].resize(maxNodes);
        for(j=0;j<maxNodes;j++)
            M[i][j]=0;
    }
	for(i=0;i<(int)hmmus.size();i++){
		for(j=0;j<(int)hmmus[i]->ins.size();j++)
            for(k=0;k<(int)hmmus[i]->outs.size();k++)
                M[hmmus[i]->ins[j]][hmmus[i]->outs[k]]=1;
	}
    return M;
}

vector<vector<int> > tAgent::getDistMap(vector<vector<int> > M){
	 int m[maxNodes][maxNodes] = {0};
	 int cdist[maxNodes][maxNodes] = {0}; // current-distance adjacency matrix (for length n these are connected...)
	 int cdistNew[maxNodes][maxNodes] = {0};
    vector<vector<int>> dist(maxNodes, vector<int>(maxNodes, 0)); // create and init 2D to 0's
    int h,i,j,k,sum,steps=1;
	 bool newRecords=false;
	 for (j=maxNodes-1; j>=0; --j) { /// initialize m and cdist arrays
		 for (k=maxNodes-1; k>=0; --k) {
			 m[j][k]=M[j][k];
			 cdist[j][k]=M[j][k];
		 }
	 }
    do {
		 newRecords=false;
		 for (j=maxNodes-1; j>=0; --j) {
			 for (k=maxNodes-1; k>=0; --k) {
				 if ((dist[j][k]==0) && (cdist[j][k]!=0)) {
					 dist[j][k]=steps;
					 newRecords=true;
				 }
			 }
		 }
		 for (j=maxNodes-1; j>=0; --j) { // dot original adjacency for new step analysis
			 for (k=maxNodes-1; k>=0; --k) {
				 sum=0;
				 for (i=maxNodes-1; i>=0; --i) {
					 sum |= m[j][i] & cdist[i][k];
				 }
				 cdistNew[j][k]=sum;
			 }
		 }
		 for (j=maxNodes-1; j>=0; --j)
			 for (k=maxNodes-1; k>=0; --k)
				 cdist[j][k]=cdistNew[j][k];
		 ++steps;
    } while(newRecords);
	 for (j=maxNodes-1; j>=0; --j) { /// zero out diagonals, otherwise they are the longest
		 dist[j][j]=0;
	 }
    return dist;
}

void tAgent::saveToDotFullLayout(const char *filename){
	FILE *f=fopen(filename,"w+t");
	size_t i,j,k;
	fprintf(f,"digraph brain {\n");
	fprintf(f,"	ranksep=2.0;\n");
	for(i=0;i<hmmus.size();i++){
		fprintf(f,"MM_%i [shape=box]\n",(int)i);
		for(j=0;j<hmmus[i]->ins.size();j++)
			fprintf(f,"	t0_%i -> MM_%i\n",hmmus[i]->ins[j],(int)i);
		for(k=0;k<hmmus[i]->outs.size();k++)
			fprintf(f,"	MM_%i -> t1_%i\n",(int)i,hmmus[i]->outs[k]);
		
	}
	fprintf(f,"}\n");
}

void tAgent::setupDots(int x, int y,double spacing){
	double xo,yo;
	int i,j,k;
	xo=(double)(x-1)*spacing;
	xo=-(xo/2.0);
	yo=(double)(y-1)*spacing;
	yo=-(yo/2.0);
	dots.resize(x*y);
	k=0;
	for(i=0;i<x;i++)
		for(j=0;j<y;j++){
//			dots[k].xPos=(double)(rand()%(int)(spacing*x))+xo;
//			dots[k].yPos=(double)(rand()%(int)(spacing*y))+yo;
			dots[k].xPos=xo+((double)i*spacing);
			dots[k].yPos=yo+((double)j*spacing);
//			cout<<dots[k].xPos<<" "<<dots[k].yPos<<endl;
			k++;
		}
}

void tAgent::saveLogicTable(FILE *f){
	int i,j;
	fprintf(f,"0_t0,1_t0,2_t0,3_t0,4_t0,5_t0,6_t0,7_t0,8_t0,9_t0,10_t0,11_t0,12_t0,13_t0,14_t0,15_t0,,0_t1,1_t1,2_t1,3_t1,4_t1,5_t1,6_t1,7_t1,8_t1,9_t1,10_t1,11_t1,12_t1,13_t1,14_t1,15_t1\n");
	for(i=0;i<65536;i++){
		for(j=0;j<16;j++){
			fprintf(f,"%i,",(i>>j)&1);
			states[j]=(i>>j)&1;
		}
		updateStates();
		for(j=0;j<16;j++){
			fprintf(f,",%i",states[j]);
		}
		fprintf(f,"\n");
	}
}
void tAgent::saveGenome(FILE *f){
	size_t i;
	for(i=0;i<genome.size();i++)
		fprintf(f,"%i	",genome[i]);
	fprintf(f,"\n");
}




