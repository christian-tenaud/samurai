// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include <CLI/CLI.hpp>

#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include "Roe_scheme.hpp"

#include <algorithm>
#include <ctime>

#include <filesystem>
namespace fs = std::filesystem;

template <class xtensor_t, class xtensor_c, std::size_t order, std::size_t field_size>
auto compute_OS_flux_correction(xtensor_t& d_alpha, xtensor_t& nu, xtensor_c& c_order, std::size_t j)
{
    //using value_type = typename xtensor_t::value_type;

    static constexpr double zero = 1e-14;

    // std::cout << " compute_OS : d_alpha  = " << d_alpha << std::endl;
    // std::cout << " compute_OS : nu  = " << nu << std::endl;
    // std::cout << " compute_OS : c_order = " << c_order << std::endl;

    //value_type flux;
    xt::xtensor_fixed<double, xt::xshape<field_size>> flux;

    //Accuracy function for each k-wave
    for (std::size_t l = 0; l < field_size; ++l)
    {

        // Lax-Wendroff
        double phi_o = (1.-nu[j](l))*d_alpha[j](l); 

        // 3rd order
        if( order >= 2)
        {
            phi_o += - c_order(0,j)(l)*(1.-nu[j](l))*d_alpha[j](l)
                     + c_order(0,j-1)(l) * (1.-nu[j-1](l))*d_alpha[j-1](l);
        }

       if( order >= 3)
       {
           // 4th order       
           phi_o += c_order(1,j)(l)*(1.-nu[j](l))*d_alpha[j](l) 
                  - 2.*c_order(1,j-1)(l)*(1.-nu[j-1](l))*d_alpha[j-1](l)
                  + c_order(1,j-2)(l)*(1.-nu[j-2](l))*d_alpha[j-2](l);
           // 5th order
           phi_o += - ( c_order(2,j+1)(l)*(1.-nu[j+1](l))*d_alpha[j+1](l)
                      - 3.*c_order(2,j)(l)*(1.-nu[j](l))*d_alpha[j](l)
                      + 3.*c_order(2,j-1)(l)*(1.-nu[j-1](l))*d_alpha[j-1](l)
                      - c_order(2,j-2)(l)*(1.-nu[j-2](l))*d_alpha[j-2](l) );
       }

        if( order >= 4)
        {
           // 6th order
           phi_o += c_order(3,j+2)(l)*(1.-nu[j+2](l))*d_alpha[j+2](l)
                  - 4.*c_order(3,j+1)(l)*(1.-nu[j+1](l))*d_alpha[j+1](l)
                  + 6.*c_order(3,j)(l)*(1.-nu[j](l))*d_alpha[j](l)
                  - 4.*c_order(3,j-1)(l)*(1.-nu[j-1](l))*d_alpha[j-1](l)
                  + c_order(3,j-2)(l)*(1.-nu[j-2](l))*d_alpha[j-2](l);
           // 7th order
           phi_o += - ( c_order(4,j+2)(l)*(1.-nu[j+2](l))*d_alpha[j+2](l)
                      - 5.*c_order(4,j+1)(l)*(1.-nu[j+1](l))*d_alpha[j+1](l)
                      + 10.*c_order(4,j)(l)*(1.-nu[j](l))*d_alpha[j](l)
                      - 10.*c_order(4,j-1)(l)*(1.-nu[j-1](l))*d_alpha[j-1](l)
                      + 5.*c_order(4,j-2)(l)*(1.-nu[j-2](l))*d_alpha[j-2](l)
                      - c_order(4,j-3)(l)*(1.-nu[j-3](l))*d_alpha[j-3](l) );
        }

        phi_o = phi_o / ((1-nu[j](l))*d_alpha[j](l) + zero);

        //TVD constraints
        
        double r = (1-nu[j-1](l))*(d_alpha[j-1](l) + zero) / ((1-nu[j](l))*d_alpha[j](l) + zero);
        double phi_lim = std::max(0., std::min(2./(1-nu[j](l)+zero), std::min(phi_o, 2.*r/(nu[j-1](l)+zero))));

//        std::cout << " phi_lim = " << phi_lim << std::endl;
        
        // phi_lim = 0.;

        flux(l) = (1. - phi_lim * (1-nu[j](l))) * d_alpha[j](l);
    }

    return flux;
}

template <class Field, std::size_t dir, std::size_t order>
auto make_OS_scheme(double& dt)
{
    static constexpr std::size_t dim               = Field::dim;
    static constexpr std::size_t field_size        = Field::size;
    static constexpr std::size_t output_field_size = field_size;
    static constexpr std::size_t stencil_size      = 2*order;

    static constexpr double gamma  = 1.4;

    //static_assert(dim == field_size || field_size == 1,
    //            "make_OS_scheme() is not implemented for this field size in this space dimension.");

    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    samurai::FluxDefinition<cfg> lw;

    samurai::static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
        [&](auto integral_constant_d)
        {
            static constexpr int d = decltype(integral_constant_d)::value;

            if constexpr (d == dir) 
            {

                auto f = [](auto u) -> samurai::FluxValue<cfg>
                {
                    double pressure = compute_Pressure<samurai::FluxValue<cfg>, dim, field_size>(u, gamma);
                    double enthalpy = compute_Enthalpy<samurai::FluxValue<cfg>, dim, field_size>(u, gamma);

                    samurai::FluxValue<cfg> flux;
                    flux[0] = u(d+1);

                    for (std::size_t l = 1; l < dim+1; ++l)
                    {
                        flux[l] = u(d+1)*u(l)/u(0);
                    }                    
                    flux[d+1] = flux[d+1] + pressure;
                    flux[field_size-1] = u(d+1)*enthalpy;

                    return  flux;
                };

                lw[d].stencil = samurai::line_stencil_from<dim, d, stencil_size>(1-static_cast<int>(order));

                lw[d].cons_flux_function = [f, &dt](auto& cells, const Field& u) -> samurai::FluxValue<cfg>
                {
                    static constexpr std::size_t j = order-1;

                    xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size>> uj;
                    for (std::size_t l = 0; l < stencil_size; ++l)
                    {    
                        uj[l] = u[cells[l]];
                    }
                    
                    // Roe mean values
                    xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size-1>> ujp12;
                    for (std::size_t l = 0; l < stencil_size-1; ++l)
                    {    
                        ujp12[l] = compute_Roemean<samurai::FluxValue<cfg>, dim, field_size>(uj[l], uj[l+1], gamma);
                    }

                    // EigenValues
                    xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size-1>> lambda;
                    for (std::size_t l = 0; l < stencil_size-1; ++l)
                    {    
                        lambda[l] = compute_EigenValues<samurai::FluxValue<cfg>, dim, field_size>(ujp12[l], d, gamma);
                    }
                    
                    // EigenVectors
                    xt::xtensor_fixed<xt::xtensor_fixed<double, xt::xshape<field_size, field_size>>, xt::xshape<stencil_size-1>> L_jp12;
                    for (std::size_t l = 0; l < stencil_size-1; ++l)
                    {    
                        L_jp12[l] = compute_LeftEigenVectors<samurai::FluxValue<cfg>, dim, field_size>(ujp12[l], d, gamma);
                    }

                    xt::xtensor_fixed<xt::xtensor_fixed<double, xt::xshape<field_size, field_size>>, xt::xshape<stencil_size-1>> R_jp12;
                    for (std::size_t l = 0; l < stencil_size-1; ++l)
                    {    
                        R_jp12[l] = compute_RightEigenVectors<samurai::FluxValue<cfg>, dim, field_size>(ujp12[l], d, gamma);
                    }

                    auto dx = cells[j].length;

                    // Flux value centered at the interface j+1/2
                    samurai::FluxValue<cfg> flux_euler;
                    flux_euler = 0.5 * ( f(uj[j]) + f(uj[j+1]) ); 

                    // Calculation of flux correction for each k-wave
                    xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size-1>> nu;
                    xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size-1>> delta_u;
                    xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size-1>> d_alpha;

                    for (std::size_t k = 0; k < field_size; ++k)
                    {
                        if (lambda[j](k) >= 0)
                        {
                            for (std::size_t l = 0; l < stencil_size-1; ++l)
                            {   
                                nu[l](k)  = (dt / dx) * std::abs(lambda[l](k));
                            }

                            for (std::size_t l = 0; l < stencil_size-1; ++l)
                            {
                                delta_u[l] = uj[l+1] - uj[l];
                            }

                            // Riemann Invariants: delta_w = L * delta_u
                            for (std::size_t l = 0; l < stencil_size-1; ++l)
                            {   
                                d_alpha[l](k) = 0.;
                                for (std::size_t m = 0; m < field_size; ++m)
                                {
                                    d_alpha[l](k) += L_jp12[l](k,m) * delta_u[l](m);
                                }
                            }
                        }
                        else
                        {
                            for (std::size_t l = 0; l < stencil_size-1; ++l)
                            {   
                                nu[l](k)  = (dt / dx) * std::abs(lambda[stencil_size-2-l](k));
                            }

                            for (std::size_t l = 0; l < stencil_size-1; ++l)
                            {
                                delta_u[l] = uj[l+1] - uj[l];
                            }

                            // Riemann Invariants: delta_w = L * delta_u
                            for (std::size_t l = 0; l < stencil_size-1; ++l)
                            {   
                                d_alpha[l](k) = 0.;
                                for (std::size_t m = 0; m < field_size; ++m)
                                {
                                    d_alpha[l](k) += L_jp12[stencil_size-2-l](k,m) * delta_u[stencil_size-2-l](m);
                                }
                            }
                        }
                    }
                    
                    //coefficients for high-order approximations up to 7th-order
                    xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<5, stencil_size-1>> c_order;
                    for (std::size_t l = 0; l < stencil_size-1; ++l)
                    {
                        c_order(0,l) = (1.+nu[l])/3.;
                        c_order(1,l) = c_order(0,l) * (nu[l]-2)/4.;
                        c_order(2,l) = c_order(1,l) * (nu[l]-3)/5.;
                        c_order(3,l) = c_order(2,l) * (nu[l]+2)/6.;
                        c_order(4,l) = c_order(3,l) * (nu[l]+3)/7.;
                    }
                    
                    //Euler Flux correction
                    samurai::FluxValue<cfg> flux_corr;
                    
                    flux_corr = compute_OS_flux_correction<decltype(d_alpha), decltype(c_order), order, field_size>(d_alpha, nu, c_order, j);

                    // For each k-wave
                    for (std::size_t k = 0; k < field_size; ++k)
                    {
                        for (std::size_t m = 0; m < field_size; ++m)
                        {
                            flux_euler(k) -= 0.5 * R_jp12[j](k,m) * std::abs(lambda[j](m)) * flux_corr(m);
                        }
                    }

                    return flux_euler;
                };

            }
            else 
            {
                lw[d].cons_flux_function = [](auto& , const Field& ) -> samurai::FluxValue<cfg>
                {
                    samurai::FluxValue<cfg> flux;
                    flux.fill(0);
                    return flux;
                };
            }
        });

    return samurai::make_flux_based_scheme(lw);
}


template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix = "")
{
    auto mesh   = u.mesh();
    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::for_each_cell(mesh, [&](const auto& cell)
                            {
                                level_[cell] = cell.level;
                            });

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u, level_);
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    constexpr std::size_t dim = 2; 
    constexpr std::size_t field_size = dim+2;
    
   // Order of the scheme:  ordre = 2*norder-1
    constexpr std::size_t norder = 4;
    //ghost width
    constexpr std::size_t ghost_width = norder;    
    //graduation width
    constexpr std::size_t graduation_width = norder;    
    //prediction order
    constexpr std::size_t prediction_order = 1;    

    using Config              = samurai::MRConfig<dim, ghost_width, graduation_width, prediction_order>;

    using Box                 = samurai::Box<double, dim>;
    using point_t             = typename Box::point_t;

    std::cout << "------------------------- BS Vortex -------------------------" << std::endl;

    // Measure CPU time
    std::clock_t time_start = std::clock();
    
    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box  = -5;
    double right_box = 5;

    // Time integration
    double Tf  = 1.;
    double dt  = Tf / 100;
    double cfl = 0.5;

    // Multiresolution parameters
    std::size_t min_level = 1;
    std::size_t max_level = 4;
    double mr_epsilon     = 1e-3; // Threshold used by multiresolution
    double mr_regularity  = 1.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "convection_" + std::to_string(dim) + "D";
    std::size_t nfiles   = 50;
    bool export_reconstruct = false;

    CLI::App app{"Finite volume example for the heat equation in 1d"};
    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--mr-reg", mr_regularity, "The regularity criteria used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    app.add_flag("--export-reconstruct", export_reconstruct, "Export reconstructed fields")->capture_default_str()->group("Ouput");
    app.allow_extras();
    CLI11_PARSE(app, argc, argv);

    //--------------------//
    // Problem definition //
    //--------------------//
    const double pi    = std::acos(-1);
    const double gamma = 1.4;
    const double Mach  = 1.;

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    std::array<bool, dim> periodic;
    periodic.fill(true);
    samurai::MRMesh<Config> mesh{box, min_level, max_level, periodic};

    auto u    = samurai::make_field<field_size>("u", mesh);
    auto unp1 = samurai::make_field<field_size>("unp1", mesh);
    auto u1   = samurai::make_field<field_size>("u1", mesh);
    auto u2   = samurai::make_field<field_size>("u2", mesh);

    // samurai::make_bc<samurai::Neumann<nordre>>(u);
    // samurai::make_bc<samurai::Neumann<nordre>>(unp1);
    // samurai::make_bc<samurai::Neumann<nordre>>(u1);
    // samurai::make_bc<samurai::Neumann<nordre>>(u2);

    // Initial solution
    double rho_right = 1.;
    double P_right = 1.;
    double v_right = 0.;

    double rho_left = rho_right * (gamma+1.)*Mach*Mach / ( (gamma-1.)*Mach*Mach + 2. );
    double P_left = P_right * ( 2.*gamma*Mach*Mach - (gamma-1.) ) / (gamma+1.);
    
    std::cout << " Box = " << left_box << " " << right_box << std::endl;

    std::cout << " P_left = " << P_left << " rho_left = " << rho_left << std::endl;
    std::cout << " P_right = " << P_right << " rho_right = " << rho_right << std::endl;
    
    const double ampli = 5.;

    samurai::for_each_cell(mesh,
                            [&](auto& cell)
                            {
                                double dist2 = 0;
                                for (std::size_t d = 0; d < dim; ++d)
                                {
                                    dist2 += std::pow(cell.center(d), 2);
                                }
                                
                                // Balsara & Shu Vortex
                                double delta_T = -(gamma-1)*ampli*ampli*std::exp(1-dist2)/(8.*gamma*pi*pi);  
                                double t_loc   = 1. + delta_T*gamma*Mach*Mach;

                                double u_theta = 0.5*ampli*std::exp(0.5*(1-dist2)) / pi;
 
                                u[cell][0] = std::pow(t_loc, 1./(gamma-1.));
                                u[cell][1] = u[cell][0] * (1. - u_theta*cell.center(1));
                                u[cell][2] = u[cell][0] * (1. + u_theta*cell.center(0));
                                // u[cell][1] = - u[cell][0] * u_theta*cell.center(1);
                                // u[cell][2] =   u[cell][0] * u_theta*cell.center(0);
                                
                                double rho_ec = 0.;
                                for (std::size_t d = 0; d < dim; ++d)
                                {
                                    rho_ec += u[cell][d+1]*u[cell][d+1];
                                }
                                rho_ec = 0.5*rho_ec/u[cell][0];

                                u[cell][3] = u[cell][0] * t_loc / (gamma*(gamma-1.)*Mach*Mach) + rho_ec;

                                // Rankine-Hugoniot jump
                                // if( cell.center(0) <= (right_box+left_box)*.5 )
                                // {
                                //     u[cell][0] = rho_left;
                                //     u[cell][1] = rho_right * v_right;
                                //     u[cell][2] = 0.;
                                //     double rho_ec = 0.;
                                //     for ( std::size_t d = 1; d < dim+1; ++d)
                                //     {
                                //         rho_ec += u[cell][d]*u[cell][d];
                                //     }
                                //     rho_ec = 0.5*rho_ec/u[cell][0];
                                //     u[cell][field_size-1] = P_left / (gamma*(gamma-1.)*Mach*Mach) + rho_ec;   
                                    
                                // }
                                // else
                                // {
                                //     u[cell][0] = rho_right;
                                //     u[cell][1] = rho_right * v_right;
                                //     u[cell][2] = 0.;
                                //     double rho_ec = 0.;
                                //     for ( std::size_t d = 0; d < dim; ++d)
                                //     {
                                //         rho_ec += u[cell][d+1]*u[cell][d+1];
                                //     }
                                //     rho_ec = 0.5*rho_ec/u[cell][0];
                                //     u[cell][field_size-1] = P_right / (gamma*(gamma-1.)*Mach*Mach) + rho_ec;    
                                // }

                            });

    auto schemex = make_OS_scheme<decltype(u), 0, norder>(dt);
    auto schemey = make_OS_scheme<decltype(u), 1, norder>(dt);
    // auto schemez = make_OS_scheme<decltype(u), 2, norder>(dt);
    
    //--------------------//
    //   Time iteration   //
    //--------------------//

    double dx = samurai::cell_length(max_level);

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

    double dt_save    = Tf / static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;

    double sum_max_velocities = 1.;

    std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave) : "";
    save(path, filename, u, suffix);
    if (export_reconstruct) 
    {
        samurai::update_ghost_mr(u);
        auto u_recons = samurai::reconstruction(u);
        samurai::save(path, fmt::format("convection_2D_recons_ite_{}", nsave), u_recons.mesh(), u_recons);
    }
    nsave++;

    double t = 0;
    while (t != Tf)
    {
    
        //Calculate time step for stability
        sum_max_velocities = 0.;
        samurai::for_each_cell(mesh, [&](auto & cell)
        {
            double velocity = 0.;
            double ec = 0.;
            for (std::size_t l = 0; l < dim; ++l)
            {
                velocity = u[cell][l+1] / u[cell][0]; 
                ec += velocity * velocity;
            }
            ec = 0.5 * ec;
            double cson= std::sqrt( gamma * (gamma-1) * ( u[cell][field_size-1] / u[cell][0] - ec ) );

            for (std::size_t l = 0; l < dim; ++l)
            {
                velocity = std::abs( u[cell][l+1] / u[cell][0] ); 
                sum_max_velocities = std::max(sum_max_velocities, velocity + cson );
            }
        });

        double dt_CFL = 0.;
        if( sum_max_velocities != 0. )
        {
            dt_CFL = cfl * dx / sum_max_velocities;
        }
        else
        {
            dt_CFL = cfl * dx;
        }
        std::cout << "sum_max_vel = " << sum_max_velocities << std::endl;

        // Move to next timestep
        t += dt_CFL;
        if (t > Tf)
        {
            dt_CFL += Tf - t;
            t = Tf;
        }
        std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}", nt++, t, dt_CFL) << std::flush;

        // Mesh adaptation
        MRadaptation(mr_epsilon, mr_regularity);
        samurai::update_ghost_mr(u);
        u1.resize();
        u2.resize();
        unp1.resize();

        std::cout << std::endl;
        // std::cout << " Scheme X : u -> u1 " << std::endl;
        dt = 0.5 * dt_CFL;
        u1   = u  - dt * schemex(u);
        samurai::update_ghost_mr(u1);
        
        // std::cout << " Scheme Y : u1 -> u2 " << std::endl;
        dt = dt_CFL;
        u2   = u1 - dt * schemey(u1);
        samurai::update_ghost_mr(u2);

        // std::cout << " Scheme X : u2 -> unp1 " << std::endl;
        dt = 0.5 * dt_CFL;
        unp1 = u2 - dt * schemex(u2);
        samurai::update_ghost_mr(unp1);
        
        // u <-- unp1
        std::swap(u.array(), unp1.array());

        // Save the result
        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            std::cout << " nsave = " << nsave << std::endl;

            suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave) : "";
            save(path, filename, u, suffix);

            // samurai::update_ghost_mr(u);
            // auto u_recons = samurai::reconstruction(u);
            // samurai::save(path, fmt::format("burgers_2D_recons_ite_{}", nsave), u_recons.mesh(), u_recons);
            nsave++;
        }

        std::cout << std::endl;
    }

    std::clock_t time_finish = std::clock();
    double time_elapsed_ms = 1000.0 * (time_finish - time_start) / CLOCKS_PER_SEC;
    std::cout << " Iteration Number = " << nt
              << " CPU time used = " << time_elapsed_ms << " ms" << std::endl;

    // std::cout << std::endl;
    // std::cout << "Run the following command to view the results:" << std::endl;
    // std::cout << "python ../python/read_mesh.py " << filename << "_ite_ --field u level --start 0 --end " << nsave << std::endl;

    // std::cout << "MR config\n";
    // std::cout << "min level : " << min_level << std::endl;
    // std::cout << "max level : " << max_level << std::endl;
    // std::cout << "eps : " << mr_epsilon << std::endl;
    // std::cout << "dx : " << dx << std::endl;

    samurai::finalize();
    return 0;
}