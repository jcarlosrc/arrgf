// to_string do not compile, this is from internet to try to bypass the problem
#ifndef __PATCH_HPP__
#define __PATCH_HPP__

#include <sstream>
#include <iomanip>

namespace Patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
    
    std::string to_string_f(float f, int n)
    {
    	std::stringstream s;
    	s << std::fixed << std::setprecision (n) << f;
    	return s.str();
    }

    std::string to_string_f(double f, int n)
    {
    	std::stringstream s;
    	s << std::fixed << std::setprecision (n) << f;
    	return s.str();
    }

}

#endif
