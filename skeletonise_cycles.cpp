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
//typedef CGAL::Search_traits_3<Kernel> TreeTraits;
//typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
//typedef Neighbor_search::Tree Tree2;


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


// Boost directed graph cycle detection example https://www.boost.org/doc/libs/1_82_0/libs/graph/doc/file_dependency_example.html#sec:cycles
// Stack Overflow BGL directed graph cycle detection example https://stackoverflow.com/questions/41027377/does-undirected-dfs-detect-all-cycle-of-graph
//#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/undirected_dfs.hpp>

struct detect_loops : public boost::dfs_visitor<> {
    detect_loops(int& n_cycles, int root) : _n_cycles(n_cycles), _root(root) {}

    template < class Vertex, class Graph >
    void start_vertex(Vertex u, const Graph&) {
        _source = u;
    }
    //template < class Edge, class Graph > void tree_edge(Edge e, const Graph& g) { std::cout << "    (te) " << source(e, g) << " -- " << target(e, g) << std::endl; }
    template < class Edge, class Graph >
    void back_edge(Edge e, const Graph& g) {
        //std::cout << "    (be) " << source(e, g) << " -- " << target(e, g) << std::endl;
        int _target = target(e, g);
        if (_target == _source && _source == _root) {
            _n_cycles++;
        }
    }
protected:
    int _source;
    int _root;
    int& _n_cycles;
};


struct detect_branches : public boost::dfs_visitor<> {
    detect_branches(int& n_branches) : _n_branches(n_branches) {}
    template <class Vertex, class Graph>
    void initialize_vertex(Vertex u, const Graph& g) {
        std::cout << "initialize_vertex " << u << std::endl;
    }
    template <class Vertex, class Graph>
    void start_vertex(Vertex u, const Graph& g) {
      //  _source = u;
      //  std::cout << "start_vertex " << u << std::endl;
    }
    template < class Edge, class Graph > void tree_edge(Edge e, const Graph& g) {
        std::cout << "    (tree_edge) " << source(e, g) << " -- " << target(e, g) << std::endl;
    }
    template < class Edge, class Graph >
    void back_edge(Edge e, const Graph& g) {
        std::cout << "    (back_edge) " << source(e, g) << " -- " << target(e, g) << std::endl;
        int _target = target(e, g);
        if (_target == _source) {
            _n_branches++;
        }
    }
protected:
    int _source;
    int& _n_branches;
};


/*


template <class Visitors = null_visitor>
class dfs_visitor {
public:
    dfs_visitor() { }
    dfs_visitor(Visitors vis) : m_vis(vis) { }

    template <class Vertex, class Graph>
    void initialize_vertex(Vertex u, const Graph& g) {
        invoke_visitors(m_vis, u, g, ::boost::on_initialize_vertex());
    }
    template <class Vertex, class Graph>
    void start_vertex(Vertex u, const Graph& g) {
        invoke_visitors(m_vis, u, g, ::boost::on_start_vertex());
    }
    template <class Vertex, class Graph>
    void discover_vertex(Vertex u, const Graph& g) {
        invoke_visitors(m_vis, u, g, ::boost::on_discover_vertex());
    }
    template <class Edge, class Graph>
    void examine_edge(Edge u, const Graph& g) {
        invoke_visitors(m_vis, u, g, ::boost::on_examine_edge());
    }
    template <class Edge, class Graph>
    void tree_edge(Edge u, const Graph& g) {
        invoke_visitors(m_vis, u, g, ::boost::on_tree_edge());
    }
    template <class Edge, class Graph>
    void back_edge(Edge u, const Graph& g) {
        invoke_visitors(m_vis, u, g, ::boost::on_back_edge());
    }
    template <class Edge, class Graph>
    void forward_or_cross_edge(Edge u, const Graph& g) {
        invoke_visitors(m_vis, u, g, ::boost::on_forward_or_cross_edge());
    }
    template <class Vertex, class Graph>
    void finish_vertex(Vertex u, const Graph& g) {
        invoke_visitors(m_vis, u, g, ::boost::on_finish_vertex());
    }

    BOOST_GRAPH_EVENT_STUB(on_initialize_vertex, dfs)
        BOOST_GRAPH_EVENT_STUB(on_start_vertex, dfs)
        BOOST_GRAPH_EVENT_STUB(on_discover_vertex, dfs)
        BOOST_GRAPH_EVENT_STUB(on_examine_edge, dfs)
        BOOST_GRAPH_EVENT_STUB(on_tree_edge, dfs)
        BOOST_GRAPH_EVENT_STUB(on_back_edge, dfs)
        BOOST_GRAPH_EVENT_STUB(on_forward_or_cross_edge, dfs)
        BOOST_GRAPH_EVENT_STUB(on_finish_vertex, dfs)

protected:
    Visitors m_vis;
};
*/
int main(int argc, char** argv)
{
    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " filename scale n_scales alpha offset" << std::endl;
    }
    const std::string filename = (argc > 1) ? argv[1] : "../elephant.off";
    std::ifstream in(filename.c_str(), std::ios::binary);
    Point_set pts;
    std::cerr << "Reading input" << std::endl;
    in >> pts;
    in.close();
    if (pts.number_of_points() == 0) {
        std::cerr << "Error: no vertices found." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "No. of vertices: " << pts.number_of_points() << std::endl;

    // Step 1: Create alpha shape wrap
    const float radius_size = (argc > 2) ? atof(argv[2]) : 0.3f; // specify radius of neighborhoods (default: 60cm, MAC suggests: 10cm)
    //const float voxel_size = radius_size / 3.f; // re-scale for CGAL's feature generator
    const float voxel_size = (radius_size * 5) / 3.f; // re-scale for CGAL's feature generator (multiply by 5 to not have problems with too fine of a grid)

    // wrap surface
    // Compute the alpha and offset values
    const double relative_alpha = (argc > 4) ? std::stod(argv[4]) : 0.15f;//0.2;// 10. 
    const double relative_offset = (argc > 5) ? std::stod(argv[5]) : 2.f;//2.f;// 300.;
    std::cout << "relative alpha = " << relative_alpha << " relative offset = " << relative_offset << std::endl;
    double alpha = radius_size / relative_alpha; // bbox / relative_alpha;
    double offset = radius_size / relative_offset; // bbox / relative_offset
    alpha = 0.0666667/2.5; offset = 0.05/3;
    std::cout << "absolute alpha = " << alpha << " absolute offset = " << offset << std::endl;

    CGAL::Real_timer t;
    t.start();

    // convert to a kernel that is more stable for Alpha Wrap
    Point_container points;
    for (auto& point : pts.points()) {
        Point_Alpha pt(point.x(), point.y(), point.z());
        points.push_back(pt);
    }

    // construct the wrap
    Mesh wrap;
    CGAL::alpha_wrap_3(points, alpha, offset, wrap);
    std::cout << "Result: " << num_vertices(wrap) << " vertices, " << num_faces(wrap) << " faces, " << std::endl;
    CGAL::IO::write_polygon_mesh("wrap.ply", wrap, CGAL::parameters::stream_precision(17));

    t.stop();
    std::cout << "Took " << t.time() << " s" << std::endl;

    // Step 2: Extract mean curvature flow skeleton
    Skeleton skeleton2;
    CGAL::extract_mean_curvature_flow_skeleton(wrap, skeleton2);

    Skeleton skeleton(10);
    boost::add_edge(0, 1, skeleton);
    boost::add_edge(1, 2, skeleton);
    boost::add_edge(2, 3, skeleton);
    boost::add_edge(3, 4, skeleton);
    boost::add_edge(4, 5, skeleton);
    boost::add_edge(5, 6, skeleton);
    boost::add_edge(6, 7, skeleton);
    boost::add_edge(7, 8, skeleton);

    std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << std::endl;
    std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << std::endl;

    // find cycles in skeleton graph
    std::map<Skeleton_edge, boost::default_color_type> edge_color;
    auto ecmap = boost::make_assoc_property_map(edge_color);

    std::vector<int> cycles;
    cycles.reserve(boost::num_vertices(skeleton));
    for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
        int n_cycles = 0;
        detect_loops vis(n_cycles, v);
        boost::undirected_dfs(skeleton, boost::root_vertex(v).visitor(vis).edge_color_map(ecmap));
        cycles.push_back(n_cycles);
    }

    std::map<Skeleton_edge, boost::default_color_type> edge_color2;
    auto ecmap2 = boost::make_assoc_property_map(edge_color2);

    std::vector<int> branches;
    branches.reserve(boost::num_vertices(skeleton));
    for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
        int n_branches = 0;
        detect_branches vis(n_branches);
        boost::undirected_dfs(skeleton, boost::root_vertex(v).visitor(vis).edge_color_map(ecmap2));
        branches.push_back(n_branches);
    }

    std::cout << "Branches: ";
    for (int branch : branches) {
        std::cout << branch << " ";
    }
    std::cout << std::endl;


    Point_set pts2;
    for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton)))
        pts2.insert(Point(skeleton[v].point.x(), skeleton[v].point.y(), skeleton[v].point.z()));

    Fmap branchmap = pts2.add_property_map<float>("scalar_Branch", 0).first;
    for (int i = 0; i < pts2.number_of_points(); ++i) {
        branchmap[i] = (float)branches.at(i);
    }

    std::ofstream f_branches("branches_output.ply");
    f_branches.precision(18);
    f_branches << pts2;


    // write skeleton w/ cycle counters to a file
    Point_set pts1;
    for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton)))
        pts1.insert(Point(skeleton[v].point.x(), skeleton[v].point.y(), skeleton[v].point.z()));

    Fmap cyclemap = pts1.add_property_map<float>("scalar_Cycle", 0).first;
    for (int i = 0; i < pts1.number_of_points(); ++i) {
        cyclemap[i] = (float) cycles.at(i);
    }

    std::ofstream f("skel_points.ply");
    f.precision(18);
    f << pts1;
  
    std::cerr << "All done" << std::endl;
    return EXIT_SUCCESS;
}