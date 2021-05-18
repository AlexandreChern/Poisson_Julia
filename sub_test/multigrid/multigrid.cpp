#include "Array.h"
#include <iostream>
using namespace std;

#define RES(i,j) (RHS[lev](i,j) + invh2[lev]*(Sol[lev](i+1.,j)
        + Sol[lev](i,j+1) + Sol[lev](i,j-1)
        + Sol[lev](i-1,j))
        - (4.0*invh2[lev] + alpha) * Sol[lev](i,j))

double alpha = double (0.0001);

// grid level factro 1/h^2

double *invh2;

int main (int argc, char** argv)
{
    if (argc != 5)
    {
        cout << "Args: nrows ncols levels cycles" << endl;
        exit (0);
    }

    int nrows = atoi (argv[1]);
    int ncols = atoi (argv[2]);
    int nlevels = atoi (argv[3]);
    int iters = atoi (argv[4]);

    // Solution and right hand side on each level
    Array<double>* Sol = new Array<double>[nlevels];
    Array<double>* RHS = new Array<double>[nlevels];

    // add 1 ghost layer in each direction
    int sizeadd = 2;
    invh2 = new doubl[nlevels];


    // allocate memory on each grid level
    for (int i = 0; i < nlevels; i++){
        Sol[i].resize (nrows + sizeadd, ncols + sizeadd);
        RHS[i].resize (nrows + sizeadd, ncols + sizeadd);
        nrows = ( nrows / 2 );
        ncols = ( ncols / 2 );
        invh2[i] = double (1.0)
                    / (pow (double (4.0), i)); 
    }

    // initialize solution and right hand sice
    Sol[nlevels-1].initrandom(0,1);
    Sol[0] = 1;

    // simply set f = 0 to obtain the solution u = 0
    RHS[0] = 0;

    // restrict right hand side to all coarse grids
    for (int i = 1; i < nlevels; i++){
        Restrict (RHS[i-1], RHS[i])
    }

    // compute normalized starting residual
    double res_0 = residual (0, Sol, RHS);
    cout << "Starting residual: " << res_0 << endl;

    // call FMG solver
    cout << "Residual after FMG iteration: " << FMG_Solver(Sol, RHS, nlevels, iters, res0) << endl;

    delete [] invh2;
    delete [] Sol;
    delete [] RHS;

    return 0;
}



double FMG_Solver(Array<double>* Sol, Array<double>* RHS, int nlevels, int iters, double res_0)
{
    double res = res_0, res_old;
    // no. of pre-, post-, and coarse smoothing steps
    int nprae = 2, npost = 1, ncoarse = 10;

    for (int lev = nlevels-1; lev >= 0; lev--)
    {
        // do a fixed no. of V-cycles on each level
        for (int i = 0; i < iters; i++){
            res_old = res;
            VCycle (lev, Sol, RHS, nprae, npost, nlevels);

            // compute residual as accuracy measure
            res = residual (lev, Sol, RHS);

            cout << "FMG level: " << lev << " cycle: "
                << i << " residual: " << res
                << " residual reduction: " << res_0/res
                << " convergence factor: "
                << res/res_old << endl;
        }

        // if not finiest level in interpolate current solution
        if (lev > 0 )
            interpolate (Sol[lev-1], Sol[lev], lev);
    }
    return res;
}


void VCycle (int lev, Array<double>* Sol, Array<double>* RHS, 
            int nprae, int npost, int ncoarse, int nlevels ) 
{
    // solve problem on coarsest grid ...
    if (lev == nlevels - 1)
        for (int i = 0; i < ncoarse; i++)
            GaussSeidel (lev, Sol, RHS);
    else
    // ... or recursively do V-cycle
    {
        // do some presmoothing steps
        for (int i = 0; i < nprae; i++)
            GaussSeidel (lev, Sol, RHS);

        // compute and restrict the residual
        Restrict_Residual (lev, Sol, RHS);

        // initialzie the coarse solution to zero
        Sol[lev+1] = 0;

        VCycle(lev+1, Sol, RHS, nprae, npost, ncoarse, nlevels) ;

        // interpolate error and correct fine solution
        interpolate_correct (Sol[lev], Sol[lev+1], lev+1);

        // do some postsmoothing steps
        for (int i = 0; i < npost; i ++)
            GaussSeidel (lev, Sol, RHS);
    }
}


void GaussSeidel (int lev, Array<double>* RHS )
{
    double denom = double (1.0)
                / (double (4.0) * invh2[lev] + alpha);
    // assure boundary conditions
    treatboundary (lev, Sol);

    // Gauss-Seidel relaxation with damping parameters
    for (int i = 1; i < Sol[lev].nrows()-1; i++)
    {
        for (int j = 1; j < Sol[lev].ncols()-1; j++)
        {
            Sol[lev] (i,j) = (RHS[lev] (i,j) + 
                invh2[lev] * (Sol[lev](i+1,j) + Sol[lev](i-1,j) + 
                Sol[lev](i,j+1) + Sol[lev](i,j-1))
                ) * denom;
        }
    }

    // assure boundary conditions
    treatboundary (lev, Sol);
}


void treatboundary (int lev, Array<double>* Sol)
{
    // treat left and right boundary
    for (int i=0; i < Sol[lev].nrows(); i++)
    {
        Sol[lev](i,0) = Sol[lev] (i,1);
        Sol[lev](i, Sol[lev].ncols() - 1) = 
                Sol[lev] (i, Sol[lev].ncols() - 2);
    }

    // treat upper and lower boundary
    for (int j=0; j < Sol[lev].ncols(); j++)
    {
        Sol[lev] (0,j) = Sol[lev] (i,j);
        Sol[lev] (Sol[lev].nrows()-1,j) = 
            Sol[lev] (Sol[lev].nrows()-2,j);
    }
}

double residual (int lev, Array<double>* Sol, Array<double>* RHS)
{
    double res = double (0.0);

    double rf;

    for (int i = 1; i < Sol[lev].nrows() - 1; i++)
    {
        for (int j = 1; j < Sol[lev].ncols() - 1; j++)
        {
            rf = RES ( i, j );
            res += rf * rf;
        }
    }
    return sqrt (res) 
            / (Sol[lev].nrows() * Sol[lev].ncols());

}

void Restrict (Array<double> & fine, Array<double> & coarse)
{
    // loop over coarse grid points
    for (int i = 1; i < coarse.nrows()-1; i++)
    {
        int fi = 2*i;
        for (int j = 1; j < coarse.ncols()-1; j++)
        {
            int fj = 2*j
            coarse (i,j) = double (0.25) * 
                (fine (fi, fj) + fine (fi - 1, fj) +
                fine (fi,fj-1) + fine (fi-1,fj-1));
        }
    }
}

void Restrict_Residual (int lev, Array<double>* Sol, 
                Array<double>* RHS)
{
    // loop over coarse grid points
    for (int i = 1; i < RHS[lev+1].nrows()-1; i++)
    {
        int fi = 2*i;
        for (int j = 1; j < RHS[lev+1].ncols()-1; j++)
        {
            int fj = 2*j
            RHS[lev+1] (i,j) = double (0.25) * 
            (RES(fi,fj) + RES(fi-1,fj) +
            RES(fi,fj-1) + RES(fi-1,fj-1));
        }
    }
}


void interpolate (Array<double> & uf, Array<double>& uc, int l)
{
    double v;

    // loop over coarse grid points
    for (int i = 1; i < uc.nrows() - 1; i++)
    {
        int fi = 2*i;
        for (int j = 1; j < uc.ncols() - 1; j++)
        {
            int fj = 2*j
            v = uc(i,j);

            uf (fi,fj) = v;
            uf (fi-1,fj) = v;
            uf (fi,fj-1) = v;
            uf (fi-1,fj-1) = v;
        }
    }
}

void interpolate_correct (Array<double>& uf,
                        Array<double>& uc, int l)
{
    double v;
    
    // loop over coarse grid points
    for (int i=1; i < uc.nrows()-1;i++)
    {
        int fi = 2*i;
        for (int j=1;j < uc.ncols()-1;j++)
        {
            int fj = 2*j;
            v = uc(i,j);

            uf (fi,fj) += v;
            uf (fi-1;fj) += v;
            uf (fi,fj-1) += v;
            uf (fi-1,fj-1) += v;
        }
    }
}