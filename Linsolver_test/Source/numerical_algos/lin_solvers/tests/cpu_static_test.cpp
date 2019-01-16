
#include <iostream>
#include <cmath>
#include "LinSolvCGS.h"

class	vec
{
public:
	double x,y;
};

class	vec_ops_t
{
public:
	void verify_size(vec&x)const {}
	//calc: B = B*mulB + A*mulA
	void addMul(vec& B, double mulB, const vec& A, double mulA)const
	{
		B.x = B.x*mulB + A.x*mulA;
		B.y = B.y*mulB + A.y*mulA;
	}
	//calc: C = C*mulC + A*mulA + B*mulB
	void addMul(vec& C, double mulC, const vec& A, double mulA, const vec& B, double mulB)const
	{
		C.x = C.x*mulC + A.x*mulA + B.x*mulB;
		C.y = C.y*mulC + A.y*mulA + B.y*mulB;
	}
	double	norm(const vec& x)const
	{
		return std::sqrt(x.x*x.x + 0.1*x.y*x.y);
	}
	double	scalar_prod(const vec& x, const vec& y)const
	{
		return x.x*y.x + x.y*y.y;
	}
};

class op_t
{
public:
	double	a00,a01,a10,a11;
	void apply(const vec& x,vec& f)const
	{
		f.x = x.x * a00 + x.y * a01;
		f.y = x.x * a10 + x.y * a11;
	}
};

class prec_t
{
public:
	void apply(vec& x)const
	{
		//DO nothing
	}
};

typedef LinSolvCGS<op_t,prec_t,vec_ops_t,vec,double> solv_t;

int main()
{
	solv_t		solv;
	vec_ops_t	vec_ops;
	op_t		op;
	prec_t		P;
	vec		x, b;
	x.x = 10.f;
	x.y = 2.f;

	b.x = -5.f;
	b.y = 10.f;

	op.a00 = 1.f;  op.a01 = 2.f;
	op.a10 = 3.f;  op.a11 = 4.f;

        solv.abs_tol = 1e-10;

	solv.solve(vec_ops, op, P, x, b);
        std::cout << x.x << " " << x.y << std::endl;

	return 0;
}