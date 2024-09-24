#if defined (_MSC_VER) && !defined (_WIN64)
#pragma warning(disable:4244) // boost::number_distance::distance()
// converts 64 to 32 bits integers
#endif

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cmath>

#define LOG() std::cout<<__FILE__<<":"<<__LINE__<< std::endl;

#include <string>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Classification.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Real_timer.h>
#include <CGAL/Classification/ETHZ/Random_forest_classifier.h>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Point_set_3<Point> Point_set;
typedef std::vector<Point> Point_container2;
typedef Kernel::Iso_cuboid_3 Iso_cuboid_3;
//typedef Point_set::Point_map Pmap;
typedef Point_set::Property_map<int> Imap;
typedef Point_set::Property_map<unsigned char> UCmap;
typedef Point_set::Property_map<float> Fmap;
typedef CGAL::Identity_property_map<Point> Pmap;

namespace Classification = CGAL::Classification;
typedef Classification::Label_handle                                            Label_handle;
typedef Classification::Feature_handle                                          Feature_handle;
typedef Classification::Label_set                                               Label_set;
typedef Classification::Feature_set                                             Feature_set;
typedef Classification::Point_set_feature_generator<Kernel, Point_set, Pmap>    Feature_generator;

#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Classification/Point_set_neighborhood.h>
#include <CGAL/IO/write_ply_points.h>
#include <utility>

typedef CGAL::Classification::Point_set_neighborhood<Kernel, Point_container2, Pmap>             Neighborhood;
typedef Neighborhood::K_neighbor_query KNeighborQuery;


#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mean_curvature_flow_skeletonization.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>


namespace AW3 = CGAL::Alpha_wraps_3;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel_Alpha;
typedef Kernel_Alpha::Point_3 Point_Alpha;
using Point_container = std::vector<Point_Alpha>;
using Mesh = CGAL::Surface_mesh<Point_Alpha>;
typedef CGAL::Mean_curvature_flow_skeletonization<Mesh> Skeletonization;
typedef Skeletonization::Skeleton                             Skeleton;
typedef Skeleton::vertex_descriptor                                  Skeleton_vertex;
typedef Skeleton::edge_descriptor                             Skeleton_edge;

typedef boost::graph_traits<Mesh>::vertex_descriptor    vertex_descriptor;

typedef CGAL::Search_traits_3<Kernel_Alpha> Traits;
typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_sphere;
typedef CGAL::Kd_tree<Traits> Tree;


typedef boost::graph_traits<Mesh>::vertices_size_type size_type;
typedef boost::property_map<Mesh, CGAL::vertex_point_t>::type Vertex_point_pmap;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>                      K_neighbor_search;
typedef K_neighbor_search::Tree                                         Tree;
typedef Tree::Splitter                                                  Splitter;
typedef K_neighbor_search::Distance                                     Distance;


#include <boost/graph/undirected_dfs.hpp>

struct detect_loops : public boost::dfs_visitor<> {
    detect_loops(int& n_cycles) : _n_cycles(n_cycles) {}

    template < class Vertex, class Graph >
    void start_vertex(Vertex u, const Graph&) {
        _source = u;
        std::cout << "    (root) " << _source << std::endl;
    }
    template < class Edge, class Graph > void tree_edge(Edge e, const Graph& g) { std::cout << "    (te) " << source(e, g) << " -- " << target(e, g) << std::endl; }
    template < class Edge, class Graph >
    void back_edge(Edge e, const Graph& g) {
        std::cout << "    (be) " << source(e, g) << " -- " << target(e, g) << std::endl;
        int _target = target(e, g);
        if (_target == _source) {
            _n_cycles++;
            std::cout << "    cycle found (" << _source << ", " << _target << ")" << std::endl;
        }
    }
protected:
    int _source;
    int& _n_cycles;
};

struct detect_branches : public boost::dfs_visitor<> {
    detect_branches(int& branches, const std::map<Skeleton_vertex, int>& cycles) : _branches(branches) , _cycles(cycles) {}

    template < class Vertex, class Graph >
    void start_vertex(Vertex u, const Graph&) {
        _source = u;
    }
    template < class Edge, class Graph >
    void tree_edge(Edge e, const Graph& g) {
        Skeleton_vertex sd = source(e, g);
        Skeleton_vertex td = target(e, g);
        std::cout << "    (te) " << source(e, g) << " -- " << target(e, g) << " cycles(s): " << _cycles.at(sd) << " cycles(t): " << _cycles.at(td) << ", branch: " << _branch << ", branches: " << _branches << std::endl;
        if (_cycles.at(td) > 0 && _cycles.at(sd) > 0) {
            _branches =false;
        }
        if (_branch) {
            if (sd == _source) {
               _branches++;
            }
        }
    }
 /*   template < class Edge, class Graph >
    void back_edge(Edge e, const Graph& g) {
        std::cout << "    (be) " << source(e, g) << " -- " << target(e, g) << std::endl;
    }*/
protected:
    int _source;
    int &_branches;
    bool _branch = true;
    const std::map<Skeleton_vertex, int>& _cycles;
};


int main(int argc, char** argv)
{
    Skeleton skeleton(12);
    boost::add_edge(0, 1, skeleton);
    boost::add_edge(1, 2, skeleton);
    boost::add_edge(2, 3, skeleton);
    boost::add_edge(3, 4, skeleton);
    boost::add_edge(4, 0, skeleton);
    boost::add_edge(3, 5, skeleton);
    boost::add_edge(5, 4, skeleton);
    boost::add_edge(5, 6, skeleton);
    boost::add_edge(6, 7, skeleton);
    boost::add_edge(7, 8, skeleton);
    boost::add_edge(8, 12, skeleton);
    boost::add_edge(9, 10, skeleton);
    boost::add_edge(10,11, skeleton);
    

    std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << std::endl;
    std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << std::endl;

    // find cycles in skeleton graph
    std::map<Skeleton_edge, boost::default_color_type> edge_color;
    auto ecmap = boost::make_assoc_property_map(edge_color);
    std::map<Skeleton_vertex, boost::default_color_type> vertex_color;
    auto vcmap = boost::make_assoc_property_map(vertex_color);
    
    typedef typename boost::property_traits<boost::associative_property_map<std::map<Skeleton_vertex, boost::default_color_type> > >::value_type ColorValue;
    typedef boost::color_traits< ColorValue > Color;

    std::map<Skeleton_vertex, int> cycle_map;
    for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
        int n_cycles = 0;
        detect_loops vis(n_cycles);

        // mark all vertices/edges as unvisited
        typename Skeleton::vertex_iterator ui, ui_end;
        for (boost::tie(ui, ui_end) = vertices(skeleton); ui != ui_end; ++ui) {
            boost::put(vcmap, *ui, Color::white());
            vis.initialize_vertex(*ui, skeleton);
        }
        typename Skeleton::edge_iterator ei, ei_end;
        for (boost::tie(ei, ei_end) = boost::edges(skeleton); ei != ei_end; ++ei)
            boost::put(ecmap, *ei, Color::white());

        vis.start_vertex(v, skeleton);
        boost::detail::undir_dfv_impl(skeleton, v, vis, vcmap, ecmap);
        //boost::undirected_dfs(skeleton, boost::root_vertex(v).visitor(vis).edge_color_map(ecmap).vertex_color_map(vcmap));

        std::cout << "v: " << v << " no. of cycles: " << n_cycles << std::endl;
        cycle_map[v] = n_cycles;
    }


    std::map<Skeleton_vertex, int> branch_map;
    for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
        int n_branches = 0;
        detect_branches vis(n_branches, cycle_map);

        // mark all vertices/edges as unvisited
        typename Skeleton::vertex_iterator ui, ui_end;
        for (boost::tie(ui, ui_end) = vertices(skeleton); ui != ui_end; ++ui) {
            boost::put(vcmap, *ui, Color::white());
            vis.initialize_vertex(*ui, skeleton);
        }
        typename Skeleton::edge_iterator ei, ei_end;
        for (boost::tie(ei, ei_end) = boost::edges(skeleton); ei != ei_end; ++ei)
            boost::put(ecmap, *ei, Color::white());

        vis.start_vertex(v, skeleton);

        std::cout << "(branch alg) Exporing vertex: " << v << std::endl;
        boost::detail::undir_dfv_impl(skeleton, v, vis, vcmap, ecmap);

        std::cout << "v: " << v << " no. of branches: " << n_branches << std::endl;
        branch_map[v] = n_branches;
    }


    std::map<Skeleton_vertex, int> degree_map;
    for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
        degree_map[v] = boost::degree(v, skeleton);
        if (boost::degree(v, skeleton) == 1 && branch_map.at(v) == 0) 
            branch_map[v]++;
    }

    for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
        std::cout << "vertex: " << v << ", no. of cycles: " << cycle_map[v] << ", no. of branches: " << branch_map[v] << ", no. of degrees: " << degree_map[v] << std::endl;
    }
  
    std::cerr << "All done" << std::endl;
    return EXIT_SUCCESS;
}