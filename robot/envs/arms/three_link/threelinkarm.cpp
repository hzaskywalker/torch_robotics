/***************************************************
 * Automatically generated by Maple.
 * Created On: Wed Oct 16 15:39:52 2013.
***************************************************/
#ifdef WMI_WINNT
#define EXP __declspec(dllexport)
#else
#define EXP
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mplshlib.h"
static MKernelVector kv;
EXP ALGEB M_DECL SetKernelVector(MKernelVector kv_in, ALGEB args) { kv=kv_in; return(kv->toMapleNULL()); }

/***************************************************
* Variable Definition for System:

* State variable(s):
*    x[ 0] = `Main.DFPSubsys1inst.theta_arm1_R1`(t)
*    x[ 1] = diff(`Main.DFPSubsys1inst.theta_arm1_R1`(t),t)
*    x[ 2] = `Main.DFPSubsys1inst.theta_arm1_R2`(t)
*    x[ 3] = diff(`Main.DFPSubsys1inst.theta_arm1_R2`(t),t)
*    x[ 4] = `Main.DFPSubsys1inst.theta_arm1_R3`(t)
*    x[ 5] = diff(`Main.DFPSubsys1inst.theta_arm1_R3`(t),t)
*
* Output variable(s):
*    y[ 0] = `Main.DFPSubsys1inst.theta_arm1_R1`(t)
*    y[ 1] = `Main.DFPSubsys1inst.theta_arm1_R2`(t)
*    y[ 2] = `Main.DFPSubsys1inst.theta_arm1_R3`(t)
*    y[ 3] = diff(`Main.DFPSubsys1inst.theta_arm1_R1`(t),t)
*    y[ 4] = diff(`Main.DFPSubsys1inst.theta_arm1_R2`(t),t)
*    y[ 5] = diff(`Main.DFPSubsys1inst.theta_arm1_R3`(t),t)
*
* Input variable(s):
*    u[ 0] = `Main.'arm1::u1'`(t)
*    u[ 1] = `Main.'arm1::u2'`(t)
*    u[ 2] = `Main.'arm1::u3'`(t)
*
************************************************/

/* Fixed parameters */
#define NDIFF 6
#define NDFA 6
#define NEQ 9
#define NPAR 0
#define NINP 3
#define NDISC 0
#define NIX1 3
#define NOUT 6
#define NCON 0
#define NEVT 0
#ifdef EVTHYST
#define NZC 2*NEVT
#else
#define NZC NEVT
#endif

typedef struct {
	double h;		/* Integration step size */
	double *w;		/* Float workspace */
	long *iw;		/* Integer workspace */
	long err;		/* Error flag */
	char *buf;		/* Error message */
} SolverStruct;

static void SolverError(SolverStruct *S, char *errmsg)
{
	sprintf(S->buf,"Error at t=%20.16e: %s\n",S->w[0],errmsg);
	if(S->err==-1) kv->error(S->buf);
	S->err=1;
}

static double dsn_zero=0.0;
static unsigned char dsn_undefC[8] = { 0, 0, 0, 0, 0, 0, 0xF8, 0x7F };
static double *dsn_undef = (double *)&dsn_undefC;
static unsigned char dsn_posinfC[8] = { 0, 0, 0, 0, 0, 0, 0xF0, 0x7F };
static double *dsn_posinf = (double *)&dsn_posinfC;
static unsigned char dsn_neginfC[8] = { 0, 0, 0, 0, 0, 0, 0xF0, 0xFF };
static double *dsn_neginf = (double *)&dsn_neginfC;
#define trunc(v) ( (v>0.0) ? floor(v) : ceil(v) )


static void DecompCInc(long n, double *A, long Ainc, long *ip)
{
	long i,j,k,m;
	double t;

	ip[n-1]=1;
	for(k=0;k<n-1;k++) {
		m=k;
		for(i=k+1;i<n;i++)
			if( fabs(A[i*Ainc+k])>fabs(A[m*Ainc+k]) ) m=i;
		ip[k]=m;
		if( m!=k ) ip[n-1]=-ip[n-1];
		t=A[m*Ainc+k]; A[m*Ainc+k]=A[(Ainc+1)*k]; A[(Ainc+1)*k]=t;
		if( t==0.0 ) { ip[n-1]=0; return; }
		t=-1.0/t;
		for(i=k+1;i<n;i++) A[i*Ainc+k]=A[i*Ainc+k]*t;
		for(j=k+1;j<n;j++) {
			t=A[m*Ainc+j]; A[m*Ainc+j]=A[k*Ainc+j]; A[k*Ainc+j]=t;
			if( t!=0.0 )
				for(i=k+1;i<n;i++) A[i*Ainc+j]+=A[i*Ainc+k]*t;
		}
	}
	if(A[(n-1)*(Ainc+1)]==0.0) ip[n-1]=0;
}
static void DecompC(long n, double *A, long *ip) { DecompCInc(n,A,n,ip); }


static void SolveCInc(long n, double *A, long Ainc, long *ip, double *b)
{
	long i,j,m;
	double t;

	if( n>1 ) {
		for(j=0;j<n-1;j++) {
			m=ip[j];
			t=b[m]; b[m]=b[j]; b[j]=t;
			for(i=j+1;i<n;i++) b[i]+=A[i*Ainc+j]*t;
		}
		for(j=n-1;j>0;j--) {
			b[j]=b[j]/A[(Ainc+1)*j];
			t=-b[j];
			for(i=0;i<=j-1;i++) b[i]+=A[i*Ainc+j]*t;
		}
	}
	b[0]=b[0]/A[0];
}
static void SolveC(long n, double *A, long *ip, double *b) { SolveCInc(n,A,n,ip,b); }


static void fp(long N, double T, double *Y, double *YP)
{
	double M[9], V[3], Z[38];
	long P[3], ti1, ti2;

	YP[0] = Y[1];
	YP[2] = Y[3];
	YP[4] = Y[5];
	for(ti1=1;ti1<=3;ti1++)
		for(ti2=1;ti2<=3;ti2++)
			M[(ti1-1)*3+ti2-1] = 0.;
	for(ti1=1;ti1<=3;ti1++)
		V[ti1-1] = 0.;
	Z[0] = sin(Y[0]);
	Z[1] = cos(Y[2]);
	Z[2] = cos(Y[4]);
	Z[3] = sin(Y[2]);
	Z[4] = sin(Y[4]);
	Z[5] = Z[1]*Z[2]-Z[3]*Z[4];
	Z[6] = cos(Y[0]);
	Z[2] = -(Z[3]*Z[2]+Z[1]*Z[4]);
	Z[4] = 0.35*(Z[6]*Z[2]-Z[0]*Z[5]);
	Z[7] = 0.35*Z[5];
	Z[8] = 1.2*Z[1];
	Z[9] = -(Z[8]+Z[7])-2.;
	Z[10] = 0.35*Z[2];
	Z[11] = Z[10]-1.2*Z[3];
	Z[12] = Z[11]*Z[6];
	Z[13] = Z[9]*Z[0]+Z[12];
	Z[2] = 0.35*(Z[6]*Z[5]+Z[0]*Z[2]);
	Z[5] = Z[9]*Z[6]-Z[11]*Z[0];
	Z[9] = Z[4]*Z[13];
	Z[11] = Z[2]*Z[5];
	M[0] = 1.+Z[9]-Z[11];
	Z[8] = -(Z[8]+Z[7])*Z[0]+Z[12];
	Z[12] = Z[0]*Z[3]-Z[6]*Z[1];
	Z[14] = -Z[2]+1.2*Z[12];
	Z[15] = Z[4]*Z[8];
	Z[16] = Z[2]*Z[14];
	M[1] = 1.+Z[15]-Z[16];
	Z[17] = -Z[2];
	Z[18] = Z[4]*Z[4];
	M[2] = 1.+Z[18]-Z[2]*Z[17];
	Z[19] = Y[8];
	Z[20] = Y[1]+Y[3];
	Z[21] = -0.6*Z[12];
	Z[22] = Y[1]+Y[3]+Y[5];
	Z[23] = Y[1]*Y[1];
	Z[22] = Z[22]*Z[22];
	Z[20] = Z[20]*Z[20];
	Z[24] = Z[20]*Z[21];
	Z[25] = Z[23]*Z[6];
	Z[26] = Z[22]*Z[2];
	Z[27] = -Z[26]-2.*(Z[25]+Z[24]);
	Z[28] = -0.6*(Z[0]*Z[1]+Z[6]*Z[3]);
	Z[20] = Z[20]*Z[28];
	Z[23] = Z[23]*Z[0];
	Z[22] = Z[22]*Z[4];
	Z[29] = -Z[22]+2.*(Z[23]-Z[20]);
	Z[30] = Z[4]*Z[27];
	Z[31] = Z[2]*Z[29];
	V[0] = Z[31]-Z[19]-Z[30];
	Z[1] = 1.8*Z[1];
	Z[32] = -(Z[1]+Z[7])-4.;
	Z[3] = Z[10]-1.8*Z[3];
	Z[10] = Z[6]*Z[3];
	Z[33] = Z[32]*Z[0]+Z[10];
	Z[3] = Z[0]*Z[3];
	Z[32] = Z[32]*Z[6]-Z[3];
	Z[5] = -(Z[5]+Z[32])*Z[21];
	Z[13] = (Z[33]+Z[13])*Z[28];
	M[3] = Z[5]+Z[13]+Z[9]-Z[11]+2.;
	Z[34] = -(Z[1]+Z[7])*Z[0]+Z[10];
	Z[12] = -Z[2]+1.8*Z[12];
	Z[8] = (Z[34]+Z[8])*Z[28];
	Z[14] = -(Z[14]+Z[12])*Z[21];
	M[4] = Z[14]+Z[8]+Z[15]-Z[16]+2.;
	M[5] = Z[18]-(Z[2]+2.*Z[21])*Z[17]+1.+2.*Z[28]*Z[4];
	Z[35] = Y[7];
	Z[24] = 3.*Z[24];
	Z[36] = -(Z[24]+Z[26])-4.*Z[25];
	Z[20] = 3.*Z[20];
	Z[37] = -(Z[20]+Z[22])+4.*Z[23];
	Z[29] = (Z[29]+Z[37])*Z[21];
	Z[27] = -(Z[36]+Z[27])*Z[28];
	V[1] = Z[29]+Z[27]+Z[31]-Z[30]-Z[35]-Z[19];
	Z[1] = -(Z[1]+Z[7])-5.;
	M[6] = -(Z[32]+Z[6]*Z[1]-Z[3])*Z[6]+Z[5]-(Z[33]+Z[0]*Z[1]+Z[10])*Z[0]+Z[13]+Z[9]-Z[11]+3.;
	M[7] = Z[14]+Z[8]+Z[15]-Z[16]+2.*(1.-Z[0]*Z[34]-Z[6]*Z[12]);
	M[8] = Z[18]-2.*(Z[0]-Z[28])*Z[4]-(Z[2]+2.*(Z[6]+Z[21]))*Z[17]+1.;
	V[2] = Z[29]+Z[27]-(Z[24]+Z[26]-Z[36]+5.*Z[25])*Z[0]-(Z[20]+Z[22]-Z[37]-5.*Z[23])*Z[6]+Z[31]-Z[30]-Y[6]-Z[35]-Z[19];
	DecompCInc(3,M,3,P);
	SolveCInc(3,M,3,P,V);
	YP[1] = V[0];
	YP[3] = V[1];
	YP[5] = V[2];
}

static void inpfn(double T, double *U)
{
	U[0] = 0.;
	U[1] = 0.;
	U[2] = 0.;
}

static void SolverUpdate(double *u, double *p, long first, long internal, SolverStruct *S)
{
	long i;

	//inpfn(S->w[0],u);
	for(i=0; i<NINP; i++) S->w[i+NDIFF+NIX1-NINP+1]=u[i];
	fp(NEQ,S->w[0],&S->w[1],&S->w[NEQ+NPAR+1]);
	if(S->w[NEQ+NPAR+1]-S->w[NEQ+NPAR+1]!=0.0) {
		SolverError(S,"index-1 and derivative evaluation failure");
		return;
	}
	if(internal) return;
}

static void SolverOutputs(double *y, SolverStruct *S)
{
	y[ 0]=S->w[ 1];
	y[ 1]=S->w[ 3];
	y[ 2]=S->w[ 5];
	y[ 3]=S->w[ 2];
	y[ 4]=S->w[ 4];
	y[ 5]=S->w[ 6];
}

static void EulerStep(double *u, SolverStruct *S)
{
	long i;

	S->w[0]+=S->h;
	for(i=1;i<=NDIFF;i++) S->w[i]+=S->h*S->w[NEQ+NPAR+i];
	SolverUpdate(u,NULL,0,0,S);
}

static void SolverSetup(double t0, double *ic, double *u, double *p, double *y, double h, SolverStruct *S)
{
	long i;

	S->h = h;
	S->iw=NULL;
	S->w[0] = t0;
	S->w[1] =  7.85398163397448279e-01;
	S->w[2] =  0.00000000000000000e+00;
	S->w[3] =  7.85398163397448279e-01;
	S->w[4] =  0.00000000000000000e+00;
	S->w[5] =  7.85398163397448279e-01;
	S->w[6] =  0.00000000000000000e+00;
	S->w[7] =  1.00000000000000000e+00;
	S->w[8] =  1.00000000000000000e+00;
	S->w[9] =  1.00000000000000000e+00;
	if(ic) for(i=0;i<NDIFF;i++) {
		S->w[i+1]=ic[i];
		S->w[i+NEQ+NPAR+1]=0.0;
	}
	SolverUpdate(u,p,1,0,S);
	SolverOutputs(y,S);
}

/*
	Parametrized simulation driver
*/
EXP long M_DECL ParamDriverC(double t0, double dt, long npts, double *ic, double *p, double *out, char *errbuf, long internal)
{
	double u[NINP],y[NOUT],w[1+2*NEQ+NPAR+NDFA+NEVT];
	long i,j;
	SolverStruct S;

	/* Setup */
	for(i=0;i<npts*(NOUT+1);i++) out[i]=*dsn_undef;
	S.w=w;
	if(internal==0) S.err=0; else S.err=-1;
	S.buf=errbuf;
	SolverSetup(t0,ic,u,p,y,dt,&S);
	/* Output */
	out[0]=t0; for(j=0;j<NOUT;j++) out[j+1]=y[j];
	/* Integration loop */
	for(i=1;i<npts;i++) {
		/* Take a step with states */
		EulerStep(u,&S);
		if( S.err>0 ) break;
		/* Output */
		SolverOutputs(y,&S);
		out[i*(NOUT+1)]=S.w[0]; for(j=0;j<NOUT;j++) out[i*(NOUT+1)+j+1]=y[j];
	}

	return(i);
}

EXP ALGEB M_DECL ParamDriver( MKernelVector kv_in, ALGEB *args )
{
	double t0,tf,dt,*ic,*p,*out;
	M_INT nargs,bounds[4],npts,naout,i;
	RTableSettings s;
	ALGEB outd;
	char buf[1000];

	kv=kv_in;
	nargs=kv->numArgs((ALGEB)args);
	if( nargs<5 || nargs>6 )
		kv->error("incorrect number of arguments");

	/* Process time vals */
	if( !kv->isNumeric(args[1]) )
		kv->error("argument #1, the initial time, must be numeric");
	t0=kv->mapleToFloat64(args[1]);
	if( !kv->isNumeric(args[2]) )
		kv->error("argument #2, the final time, must be numeric");
	tf=kv->mapleToFloat64(args[2]);
	if( t0>=tf )
		kv->error("the final time must be larger than the initial time");
	if( !kv->isNumeric(args[3]) )
		kv->error("argument #3, the time step, must be a positive numeric value");
	dt=kv->mapleToFloat64(args[3]);
	if(dt<=0)
		kv->error("argument #3, the time step, must be a positive numeric value");
	npts=(M_INT)ceil((tf+1e-10-t0)/dt)+1;

	/* Processing ic in */
	if( NDIFF==0 )
		ic=NULL;
	else if( kv->isInteger(args[4]) && kv->mapleToInteger32(args[4])==0 )
		ic=NULL;
	else if( !kv->isRTable(args[4]) ) {
		ic=NULL;
		kv->error("argument #4, the initial data, must be a 1..ndiff rtable");
	}
	else {
		kv->rtableGetSettings(&s,args[4]);
		if( s.storage != RTABLE_RECT || s.data_type != RTABLE_FLOAT64 ||
			 s.num_dimensions != 1 || kv->rtableLowerBound(args[4],1)!=1 ||
			 kv->rtableUpperBound(args[4],1) != NDIFF )
			kv->error("argument #4, the initial data, must be a 1..ndiff rtable");
		ic=(double *)kv->rtableData(args[4]);
	}

	/* Processing parameters in */
	if( NPAR==0 )
		p=NULL;
	else if( kv->isInteger(args[5]) && kv->mapleToInteger32(args[5])==0 )
		p=NULL;
	else if( !kv->isRTable(args[5]) ) {
		p=NULL;
		kv->error("argument #5, the parameter data, must be a 1..npar rtable");
	}
	else {
		kv->rtableGetSettings(&s,args[5]);
		if( s.storage != RTABLE_RECT || s.data_type != RTABLE_FLOAT64 ||
			 s.num_dimensions != 1 || kv->rtableLowerBound(args[5],1)!=1 ||
			 kv->rtableUpperBound(args[5],1) != NPAR )
			kv->error("argument #5, the parameter data, must be a 1..npar rtable");
		p=(double *)kv->rtableData(args[5]);
	}

	/* Output data table */
	if( nargs==6 ) {
		outd=NULL;
		if( !kv->isRTable(args[6]) ) {
			out=NULL;
			naout=0;
			kv->error("argument #6, the output data, must be a 1..npts,1..nout+1 C_order rtable");
		}
		else {
			kv->rtableGetSettings(&s,args[6]);
			if( s.storage != RTABLE_RECT || s.data_type != RTABLE_FLOAT64 ||
			 	s.order != RTABLE_C || s.num_dimensions != 2 ||
			 	kv->rtableLowerBound(args[6],1)!=1 ||
			 	kv->rtableLowerBound(args[6],2)!=1 ||
			 	kv->rtableUpperBound(args[6],2) != NOUT+1 )
				kv->error("argument #6, the output data, must be a 1..npts,1..nout+1 C_order rtable");
			naout=kv->rtableUpperBound(args[6],1);
			if( naout<1 )
				kv->error("argument #6, the output data, must have at least 1 output slot");
			out=(double *)kv->rtableData(args[6]);
			if(naout<npts) npts=naout;
		}
	}
	else {
		kv->rtableGetDefaults(&s);
		bounds[0]=1; bounds[1]=npts;
		bounds[2]=1; bounds[3]=NOUT+1;
		s.storage=RTABLE_RECT;
		s.data_type=RTABLE_FLOAT64;
		s.order=RTABLE_C;
		s.num_dimensions=2;
		s.subtype=RTABLE_ARRAY;
		outd=kv->rtableCreate(&s,NULL,bounds);
		out=(double *)kv->rtableData(outd);
		naout=npts;
	}
	for(i=0;i<naout*(NOUT+1);i++) out[i]=*dsn_undef;

	i=ParamDriverC(t0,dt,npts,ic,p,out,buf,1);

	/* All done */
	if(outd==NULL)
		return(kv->toMapleInteger(i));
	else
		return(outd);
}


/*  A class to contain all the information that needs to 
    be passed around between these functions, and can 
    encapsulate it and hide it from the Python interface.
    
    Written by Travis DeWolf (May, 2013)
*/
class Sim {
    /* Very simple class, just stores the variables we 
    need for simulation, and has 2 functions. Reset 
    resets the state of the simulation, and step steps it 
    forward. Tautology ftw!*/

    double* params;
    double dt, t0;
	double u0[NINP], other_out[NOUT+1], y[NOUT]; 
    double w[7 + 2 * NEQ + NPAR + NDFA + NEVT];

    SolverStruct S;
    
    public:
        Sim(double dt_val, double* params_pointer);
        void reset(double* out, double* ic);
        void step(double* out, double* u);
};

Sim::Sim(double dt_val, double* params_pointer)
{
    t0 = 0.0; // set up start time
    dt = dt_val; // set time step
    for (int i = 0; i < NINP; i++) u0[i] = 0.0; // initial control signal

    params = params_pointer; // set up parameters reference

	/* Setup */
	S.w = w;
	S.err = 0; 
}

void Sim::reset(double* out, double* ic) 
{
	SolverSetup(t0, ic, u0, params, y, dt, &S);

	/* Output */
	out[0] = t0; 
    for(int j = 0; j < NOUT; j++) {
        out[j + 1] = y[j];
    }
}

void Sim::step(double* out, double* u)
/* u: control signal */
{
    for (int k = 0; k < NOUT; k++)
        out[k] = *dsn_undef; // clear values to nan 

	/* Integration loop */
    /* Take a step with states */
    EulerStep(u, &S);

    if (S.err <= 0) 
    {
        /* Output */
        SolverOutputs(y, &S);

        out[0] = S.w[0]; 
        for(long j = 0; j < NOUT; j++) 
            out[j + 1] = y[j];
    }
}

int main (void)
{
    FILE *fd;
 
    double *ic, *p, *out;
    char* errbuf;
    long i, j, outd;
    long internal = 0;
 
    double dt = 0.00001;
 
    int time_steps = 1000000;
    double u[NINP];
    for (int k = 0; k < NINP; k++) u[k] = .1;
 
    fd = fopen("output.dat", "w");
 
    Sim sim = Sim(dt, NULL);
    sim.reset(out, NULL); // ic = NULL, use default start state
 
    for(i=0;i<time_steps;i++)
    {
        sim.step(out, u);
        fprintf(fd,"%lf ",out[0]);
        for(j=0;j<NOUT;j++)
        {
            fprintf(fd,"%lf ",out[j+1]);
        }
        fprintf(fd, "\n");
    }
 
    fclose(fd);
 
    return 0;
}
