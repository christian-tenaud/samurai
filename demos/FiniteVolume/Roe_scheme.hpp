// Copyright 2021 SAMURAI TEAM. All rights reserved.
#pragma once

// Routines to compute Euler system of equations

//////////
template <class xtensor_u, std::size_t dim, std::size_t field_size>
auto compute_Pressure(xtensor_u& uj, const double& gamma) 
{
    double rho_ec = 0.;
    for (std::size_t l = 1; l < dim+1; ++l)
    {
        rho_ec += uj(l)*uj(l);
    }
    rho_ec = 0.5*rho_ec/uj(0);

    // Pressure / (gamma * Mach**2)
    double PsgM2 = (gamma-1) * ( uj(field_size-1) - rho_ec );

    return PsgM2;
}

template <class xtensor_u, std::size_t dim, std::size_t field_size>
auto compute_SoundSpeed(xtensor_u& uj, const double& gamma) 
{
    double rho_ec = 0.;
    for (std::size_t l = 1; l < dim+1; ++l)
    {
        rho_ec += uj(l)*uj(l);
    }
    rho_ec = 0.5*rho_ec / uj(0);

    // C^2 = sqrt( gamma * Pressure / rho )
    double SoundSpeed = std::sqrt( gamma * (gamma-1)*(uj(field_size-1) - rho_ec)/ uj(0) );

    return SoundSpeed;
}

template <class xtensor_u, std::size_t dim, std::size_t field_size>
auto compute_Enthalpy(xtensor_u& uj, const double& gamma)
{
    double rho_ec = 0.;
    for (std::size_t l = 1; l < dim+1; ++l)
    {
        rho_ec += uj(l)*uj(l);
    }
    rho_ec = 0.5*rho_ec/uj(0);

    double Hj = ( gamma * uj(field_size-1) - (gamma-1) * rho_ec ) / uj(0);

    return Hj;
}

template <class xtensor_u, std::size_t dim, std::size_t field_size>
auto compute_Roemean(xtensor_u& uj, xtensor_u& ujp1, const double& gamma)
{
    double sqrt_rhoj   = std::sqrt(uj(0));
    double sqrt_rhojp1 = std::sqrt(ujp1(0));

    double Hj = compute_Enthalpy<decltype(uj), dim, field_size>(uj, gamma);
    double Hjp1 = compute_Enthalpy<decltype(ujp1), dim, field_size>(ujp1, gamma);

    xt::xtensor_fixed<double, xt::xshape<field_size>> mean_Roe;

    // density at j+1/2
    mean_Roe(0) = sqrt_rhoj * sqrt_rhojp1;                    
                    
    // momentum components at j+1/2
    for (std::size_t l = 1; l < dim+1; ++l)
    {
        mean_Roe(l) =  mean_Roe(0) * (sqrt_rhoj*uj(l)/uj(0) +  sqrt_rhojp1*ujp1(l)/ujp1(0)) / (sqrt_rhoj + sqrt_rhojp1); 
    }

    // kinetic energy at j+1/2
    double rho_ec = 0.;
    for (std::size_t l = 1; l < dim+1; ++l)
    {
        rho_ec += mean_Roe(l)*mean_Roe(l);
    }
    rho_ec = 0.5*rho_ec/mean_Roe(0);

    // Total energy at j+1/2
    double rhoH_bar = mean_Roe(0) * (sqrt_rhoj*Hj +  sqrt_rhojp1*Hjp1) / (sqrt_rhoj + sqrt_rhojp1);
    double P_bar = (gamma-1.) * (rhoH_bar - rho_ec) / gamma;
    mean_Roe(field_size-1) =  rhoH_bar - P_bar;

    return mean_Roe;
}

template <class xtensor_u, std::size_t dim, std::size_t field_size>
auto compute_EigenValues(xtensor_u& ujp12, const int& dir, const double& gamma)
{
    //double c_bar = std::sqrt(gamma*compute_Pressure<decltype(ujp12), dim, field_size>(ujp12, gamma)/ujp12(0));
    double c_bar = compute_SoundSpeed<decltype(ujp12), dim, field_size>(ujp12, gamma);
    double u_bar = ujp12(dir+1)/ujp12(0);

    xt::xtensor_fixed<double, xt::xshape<field_size>> EV;

    EV(0) = u_bar - c_bar;
    for (std::size_t l = 0; l < dim; ++l)
    {
        EV(l+1) = u_bar;
    }  
    EV(field_size-1) = u_bar + c_bar;

    return EV;
}

template <class xtensor_u, std::size_t dim, std::size_t field_size>
auto compute_LeftEigenVectors(xtensor_u& ujp12, const std::size_t& dir, const double& gamma)
{
    xt::xtensor_fixed<double, xt::xshape<field_size, field_size>> L_jp12;

    //double c_bar = std::sqrt(gamma*compute_Pressure<xtensor_u, dim, field_size>(ujp12, gamma)/ujp12(0));
    double c_bar = compute_SoundSpeed<decltype(ujp12), dim, field_size>(ujp12, gamma);
    double gm1s2u2 = 0.;
    for (std::size_t l = 0; l < dim; ++l)
    {
        gm1s2u2 += (ujp12(l+1)/ujp12(0))*(ujp12(l+1)/ujp12(0));
    }
    gm1s2u2 = 0.5*(gamma-1)*gm1s2u2;

    double oneoverc = 1./c_bar;

    if( dim == 1 )
    {
        double u_bar = ujp12(dir+1)/ujp12(0);

        L_jp12(0, 0) = 0.5 * (gm1s2u2 * oneoverc + u_bar) ;
        L_jp12(0, 1) = - 0.5 * ((gamma-1)*u_bar*oneoverc + 1.);
        L_jp12(0, 2) = (gamma-1) * 0.5 * oneoverc;

        L_jp12(1, 0) = c_bar - gm1s2u2 * oneoverc;
        L_jp12(1, 1) = (gamma-1) * u_bar * oneoverc;
        L_jp12(1, 2) = - (gamma-1) * oneoverc;

        L_jp12(2, 0) = 0.5 * (gm1s2u2 * oneoverc - u_bar) ;
        L_jp12(2, 1) = - 0.5 * ((gamma-1)*u_bar * oneoverc - 1.) ;
        L_jp12(2, 2) = (gamma-1) * 0.5 * oneoverc;
    }
    else if( dim == 2 )
    {
        std::array<double, dim> normal;
        normal.fill(0.);
        normal[dir] = 1.;
    
        double unc = 0.;
        for (std::size_t l = 0; l < dim; ++l)
        {
            unc += normal[l] * (ujp12(l+1)/ujp12(0));
        }
        unc = unc * c_bar;

        L_jp12(0,0) =  .5 * ( gm1s2u2 + unc ) * oneoverc;
        L_jp12(0,1) =  - .5 * ( (gamma-1.) * ujp12(1)/ujp12(0) * oneoverc + normal[0] );
        L_jp12(0,2) =  - .5 * ( (gamma-1.) * ujp12(2)/ujp12(0) * oneoverc + normal[1] ) ;
        L_jp12(0,3) =  (gamma-1.) * .5 * oneoverc;

        L_jp12(1,0) =  c_bar - gm1s2u2 * oneoverc;
        L_jp12(1,1) =  (gamma-1.) * ujp12(1)/ujp12(0) * oneoverc;
        L_jp12(1,2) =  (gamma-1.) * ujp12(2)/ujp12(0) * oneoverc;
        L_jp12(1,3) =  - (gamma-1.) * oneoverc;

        L_jp12(2,0) =  (normal[1] * ujp12(1)/ujp12(0) - normal[0] * ujp12(2)/ujp12(0));
        L_jp12(2,1) =  - normal[1];
        L_jp12(2,2) =    normal[0];
        L_jp12(2,3) =  0.;

        L_jp12(3,0) =  .5 * ( gm1s2u2 - unc ) * oneoverc;
        L_jp12(3,1) =  - .5 * ( (gamma-1.) * ujp12(1)/ujp12(0) * oneoverc - normal[0] ) ;
        L_jp12(3,2) =  - .5 * ( (gamma-1.) * ujp12(2)/ujp12(0) * oneoverc - normal[1] ) ;
        L_jp12(3,3) =  (gamma-1.) * .5 * oneoverc;
    }
    else if( dim == 3 )
    {
        std::cout << "L_jp12 in 3D might be implemented !!" << std::endl;
    }

    return L_jp12;
}

template <class xtensor_u, std::size_t dim, std::size_t field_size>
auto compute_RightEigenVectors(xtensor_u& ujp12, const std::size_t& dir, const double& gamma)
{
    xt::xtensor_fixed<double, xt::xshape<field_size, field_size>> R_jp12;

    double c_bar = compute_SoundSpeed<decltype(ujp12), dim, field_size>(ujp12, gamma);
    double H_bar = compute_Enthalpy<xtensor_u, dim, field_size>(ujp12, gamma);

    double oneoverc = 1./c_bar;

    double ec = 0.;
    for (std::size_t l = 0; l < dim; ++l)
    {
        ec += (ujp12(l+1)/ujp12(0)) * (ujp12(l+1)/ujp12(0));
    }
    ec = 0.5 * ec;

    if( dim == 1 )
    {
        double u_bar = ujp12(dir+1)/ujp12(0);

        R_jp12(0, 0) = oneoverc;
        R_jp12(1, 0) = (u_bar*oneoverc - 1.);
        R_jp12(2, 0) = (H_bar*oneoverc - u_bar);

        R_jp12(0, 1) = oneoverc;
        R_jp12(1, 1) = u_bar * oneoverc;
        R_jp12(2, 1) = 0.5 * u_bar * u_bar * oneoverc;

        R_jp12(0, 2) = oneoverc;
        R_jp12(1, 2) = (u_bar*oneoverc + 1.);
        R_jp12(2, 2) = H_bar*oneoverc + u_bar;
    }
    else if( dim == 2 )
    {
        std::array<double, dim> normal;
        normal.fill(0.);
        normal[dir] = 1.;
    
        double unc = 0.;
        for (std::size_t l = 0; l < dim; ++l)
        {
            unc += normal[l] * (ujp12(l+1)/ujp12(0));
        }
        unc = unc * c_bar;

        R_jp12(0,0) = oneoverc; 
        R_jp12(1,0) = ujp12(1)/ujp12(0)*oneoverc - normal[0]; 
        R_jp12(2,0) = ujp12(2)/ujp12(0)*oneoverc - normal[1]; 
        R_jp12(3,0) = (H_bar - unc) * oneoverc;

        R_jp12(0,1) = oneoverc;
        R_jp12(1,1) = ujp12(1)/ujp12(0) * oneoverc;
        R_jp12(2,1) = ujp12(2)/ujp12(0) * oneoverc;
        R_jp12(3,1) = ec * oneoverc;

        R_jp12(0,2) = 0.;
        R_jp12(1,2) = - normal[1];
        R_jp12(2,2) =   normal[0];
        R_jp12(3,2) = R_jp12(1,2) * ujp12(1)/ujp12(0) + R_jp12(2,2) * ujp12(2)/ujp12(0); 

        R_jp12(0,3) = oneoverc;
        R_jp12(1,3) = ujp12(1)/ujp12(0)*oneoverc + normal[0]; 
        R_jp12(2,3) = ujp12(2)/ujp12(0)*oneoverc + normal[1];  
        R_jp12(3,3) = (H_bar + unc) * oneoverc; 
    }
    else if( dim == 3 )
    {
        std::cout << "R_jp12 might be implemented !!" << std::endl;
    }

    return R_jp12;
}
/////////
