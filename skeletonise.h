

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
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Classification/Point_set_neighborhood.h>
#include <CGAL/IO/write_ply_points.h>
#include <utility>

#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mean_curvature_flow_skeletonization.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>


// Boost directed graph cycle detection example https://www.boost.org/doc/libs/1_82_0/libs/graph/doc/file_dependency_example.html#sec:cycles
        // Stack Overflow BGL directed graph cycle detection example https://stackoverflow.com/questions/41027377/does-undirected-dfs-detect-all-cycle-of-graph
#include <boost/graph/undirected_dfs.hpp>
/*
struct detect_loops : public boost::dfs_visitor<> {
    detect_loops(int& n_cycles) : _n_cycles(n_cycles) {}

    template < class Vertex, class Graph >
    void start_vertex(Vertex u, const Graph&) {
        _source = u;
    }
    //template < class Edge, class Graph > void tree_edge(Edge e, const Graph& g) { std::cout << "    (te) " << source(e, g) << " -- " << target(e, g) << std::endl; }
    template < class Edge, class Graph >
    void back_edge(Edge e, const Graph& g) {
        //std::cout << "    (be) " << source(e, g) << " -- " << target(e, g) << std::endl;
        int _target = target(e, g);
        if (_target == _source) {
            _n_cycles++;
        }
    }
protected:
    int _source;
    int& _n_cycles;
};*/

struct detect_loops : public boost::dfs_visitor<> {
    detect_loops(int& n_cycles) : _n_cycles(n_cycles) {}

    template < class Vertex, class Graph >
    void start_vertex(Vertex u, const Graph&) {
        _source = u;
    }
    //template < class Edge, class Graph > void tree_edge(Edge e, const Graph& g) { std::cout << "    (te) " << source(e, g) << " -- " << target(e, g) << std::endl; }
    template < class Edge, class Graph >
    void back_edge(Edge e, const Graph& g) {
        //std::cout << "    (be) " << source(e, g) << " -- " << target(e, g) << std::endl;
        int _target = target(e, g);
        if (_target == _source) {
            _n_cycles++;
        }
    }
protected:
    int _source;
    int& _n_cycles;
};

struct detect_branches : public boost::dfs_visitor<> {
    detect_branches(int& branches, const std::map<Skeleton_vertex, int>& cycles) : _branches(branches), _cycles(cycles) {}

    template < class Vertex, class Graph >
    void start_vertex(Vertex u, const Graph&) {
        _source = u;
    }
    template < class Edge, class Graph >
    void tree_edge(Edge e, const Graph& g) {
        Skeleton_vertex sd = source(e, g);
        Skeleton_vertex td = target(e, g);
        //std::cout << "    (te) " << source(e, g) << " -- " << target(e, g) << " cycles(s): " << _cycles.at(sd) << " cycles(t): " << _cycles.at(td) << ", branch: " << _branch << ", branches: " << _branches << std::endl;
        if (_cycles.at(td) > 0 && _cycles.at(sd) > 0) {
            _branches = false;
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
    int& _branches;
    bool _branch = true;
    const std::map<Skeleton_vertex, int>& _cycles;
};

namespace CGAL {

    namespace Classification {

        namespace Feature {

            typedef CGAL::Simple_cartesian<double> Kernel;
            typedef Kernel::Point_3 Point;
            typedef CGAL::Point_set_3<Point> Point_set;
            typedef std::vector<Point> Point_container2;
            typedef Kernel::Iso_cuboid_3 Iso_cuboid_3;

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


            typedef CGAL::Classification::Point_set_neighborhood<Kernel, Point_container2, Pmap>             Neighborhood;
            typedef Neighborhood::K_neighbor_query KNeighborQuery;

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


            // Fonction pour calculer la norme entre deux points
            double norme(const Point& p1, const Point& p2) {
                /*
                float dx = p2.x() - p1.x();
                float dy = p2.y() - p1.y();
                float dz = p2.z() - p1.z();
                return std::sqrt(dx * dx + dy * dy + dz * dz);
                */
                return CGAL::sqrt((p1 - p2).squared_length());
            }

            std::pair<std::vector<Point>, std::vector<float> > convert_polylines_to_point_cloud(const std::vector<std::vector<Point>>& polylines, int num_samples,const std::vector<float>& cycles) {
                std::vector<Point> point_cloud;
                std::vector<float> v1;
                int k = 0;
                for (const auto& polyline : polylines) {
                    int num_points = polyline.size();

                    if (num_points < 2) {
                        continue;
                    }

                    // Parcourir les segments de la polyline
                    for (int i = 0; i < num_points - 1; i++) {
                        const Point& prev_point = polyline[i];
                        const Point& next_point = polyline[i + 1];

                        // Échantillonner des points le long du segment et calculer les distances dans le point cloud
                        Point current_point = prev_point;
                        for (int j = 0; j < num_samples; j++) {
                            float t = static_cast<float>(j + 1) / (num_samples + 1);
                            Point interpolated_point;
                            interpolated_point = Point((1 - t) * prev_point.x() + t * next_point.x(),
                                (1 - t) * prev_point.y() + t * next_point.y(),
                                (1 - t) * prev_point.z() + t * next_point.z());

                            current_point = interpolated_point;
                            
                            point_cloud.push_back(interpolated_point);
                            v1.push_back(cycles.at(k));
                        }
                    }
                    k++;
                }
                return std::pair<std::vector<Point>, std::vector<float> >(point_cloud, v1);
            }
           /* std::vector<Point> convert_polylines_to_point_cloud(const std::vector<std::vector<Point>>& polylines, int num_samples) {
                std::vector<Point> point_cloud;
                std::vector<float> v1; 
                std::vector<float> v2;  

                for (const auto& polyline : polylines) {
                    int num_points = polyline.size();

                    if (num_points < 2) {
                        continue;
                    }

                    // Parcourir les segments de la polyline
                    for (int i = 0; i < num_points - 1; i++) {
                        const Point& prev_point = polyline[i];
                        const Point& next_point = polyline[i + 1];

                        // Calculer la distance entre les sommets des segments
                        float length = norme(prev_point, next_point);
                        v1.push_back(length); 

                        // Calculer le pas pour échantillonner les points le long du segment
                        float step_size = length / (num_samples + 1);

                        // Échantillonner des points le long du segment et calculer les distances dans le point cloud
                        Point current_point = prev_point;
                        for (int j = 0; j < num_samples; j++) {
                            float t = static_cast<float>(j + 1) / (num_samples + 1);
                            Point interpolated_point;
                            interpolated_point = Point((1 - t) * prev_point.x() + t * next_point.x(),
                                (1 - t) * prev_point.y() + t * next_point.y(),
                                (1 - t) * prev_point.z() + t * next_point.z());

                            float distance_in_point_cloud = norme(current_point, interpolated_point);
                            v2.push_back(distance_in_point_cloud);
                            current_point = interpolated_point;

                            point_cloud.push_back(interpolated_point);
                        }
                    }
                }

                // Afficher les résultats
                for (size_t i = 0; i < point_cloud.size(); i++) {
                    int segment_index = i / num_samples;
                    std::cout << "Point " << i << ": (" << point_cloud[i].x() << ", " << point_cloud[i].y() << ", " << point_cloud[i].z() << "), distance points: " << v2[i] << ", distance sommets: " << v1[segment_index]  << ", segment " << segment_index << std::endl;
                }

                return point_cloud;
            }


            */

            

            // User-defined feature
            template <typename GeomTraits, typename PointRange, typename PointMap>
            class My_skeleton : public CGAL::Classification::Feature_base
            {
                using Image_float = CGAL::Classification::Image<float>;

                using FloatMap = typename PointRange::template Property_map<float>;

                const PointRange& input;
                PointMap point_map;

                std::vector<typename FloatMap::value_type> values;

            public:
                My_skeleton(const PointRange& input, PointMap point_map, float feature_scale, const Mesh& wrap) : input(input), point_map(point_map)
                {
                    this->set_name("my_skeleton");

                        // Step 2: Extract mean curvature flow skeleton
                        // Skeleton_graph skeleton_graph = CGAL::extract_mean_curvature_flow_skeleton(wrap);

                        Skeleton skeleton;
                        CGAL::extract_mean_curvature_flow_skeleton(wrap, skeleton);
                        std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << "\n";
                        std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << "\n";


                        // Step 3: Split the skeleton into polylines
                        //Skeleton_polylines skeleton_polylines = CGAL::split_graph_into_polylines(skeleton_graph);




                        Point_set pts3;
                        for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
                            Point p(skeleton[v].point.x(), skeleton[v].point.y(), skeleton[v].point.z());
                            pts3.insert(p);
                        }
                        std::ofstream f3("skel_points.ply");
                        f3.precision(18);
                        f3 << pts3;

                        // Step 4: Convert polylines to point cloud  
                            //adjacent_vertices()
                        std::vector<std::vector<Point>> polylines;
                        for (Skeleton_edge e : CGAL::make_range(edges(skeleton))) {
                            Point_Alpha s = skeleton[source(e, skeleton)].point;
                            Point_Alpha t = skeleton[target(e, skeleton)].point;
                            //std::cout << "2 " << s << " " << t << "\n";
                            Point s1(s.x(), s.y(), s.z());
                            Point t1(t.x(), t.y(), t.z());
                            polylines.push_back(std::vector<Point>({ s1,t1 }));
                        }


                        // find cycles in skeleton graph
                        /*
                        std::map<Skeleton_edge, boost::default_color_type> edge_color;
                        auto ecmap = boost::make_assoc_property_map(edge_color);

                        std::vector<int> cycles;
                        cycles.reserve(boost::num_vertices(skeleton));
                        int il = 0;
                        for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
                            int n_cycles = 0;
                            detect_loops vis(n_cycles);
                            boost::undirected_dfs(skeleton, boost::root_vertex(v).visitor(vis).edge_color_map(ecmap));
                            //n_cycles += il++;
                            cycles.push_back(n_cycles);
                        }

                       

                        
                        std::vector<int> branches;
                        for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
                            int s = cycles.at(v);
                            //skeleton[v]
                            int no_branches = 0;
                            auto neighbors = boost::adjacent_vertices(v,skeleton);
                            for (auto vd : make_iterator_range(neighbors)) {
                                int t = cycles.at(vd);
                                //if (t == 0 || s == 0) no_branches++;
                                if (t == 0) no_branches++;
                            }
                            branches.push_back(no_branches);
                        }*/

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

                            //std::cout << "v: " << v << " no. of cycles: " << n_cycles << std::endl;
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

                            //std::cout << "(branch alg) Exporing vertex: " << v << std::endl;
                            boost::detail::undir_dfv_impl(skeleton, v, vis, vcmap, ecmap);

                            //std::cout << "v: " << v << " no. of branches: " << n_branches << std::endl;
                            branch_map[v] = n_branches;
                        }

                        {
                            Point_set pts1;
                            for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton)))
                                pts1.insert(Point(skeleton[v].point.x(), skeleton[v].point.y(), skeleton[v].point.z()));
                            Fmap cyclemap = pts1.add_property_map<float>("scalar_Cycle", 0).first;
                            for (int i = 0; i < pts1.number_of_points(); ++i) {
                                cyclemap[i] = (float)cycle_map.at(i);
                            }
                            Fmap branchmap = pts1.add_property_map<float>("scalar_Branch", 0).first;
                            for (int i = 0; i < pts1.number_of_points(); ++i) {
                                branchmap[i] = (float)branch_map.at(i);
                            }

                            std::ofstream f("skel_points123.ply");
                            f.precision(18);
                            f << pts1;
                        }

                        std::vector<float> cycles_edges;
                        for (Skeleton_edge e : CGAL::make_range(edges(skeleton))) {
                            int s = cycle_map.at(source(e, skeleton));
                            int t = cycle_map.at(target(e, skeleton));
                            float a = float(s + t) / 2;
                            cycles_edges.push_back(a);
                        }

                        std::vector<float> branches_edges;
                        for (Skeleton_edge e : CGAL::make_range(edges(skeleton))) {
                            int s = branch_map.at(source(e, skeleton));
                            int t = branch_map.at(target(e, skeleton));
                            float a = float(s + t) / 2;
                            branches_edges.push_back(a);
                        }

                        int num_samples = 10;
                        //std::vector<Point> pc = convert_polylines_to_point_cloud(polylines, num_samples);
                        std::pair<std::vector<Point>, std::vector<float> > pc_cycle_pair = convert_polylines_to_point_cloud(polylines, num_samples, branches_edges);
                        std::pair<std::vector<Point>, std::vector<float> > pc_cycle_pair2 = convert_polylines_to_point_cloud(polylines, num_samples, cycles_edges);
                        std::vector<Point> pc = pc_cycle_pair.first;
                        std::vector<float> cycle_vec = pc_cycle_pair2.second;
                        std::vector<float> branch_vec = pc_cycle_pair.second;
                        std::cout << " conversion done" << std::endl;
                        // Step 5: Construct a tree for efficient nearest neighbor queries
                            // Tree tree = construct_tree(skeleton_points);

                        {
                            Point_set pts1;
                            for (auto p : pc)
                                pts1.insert(p);

                            Fmap cyclemap = pts1.add_property_map<float>("scalar_Cycle", 0).first;
                            for (int i = 0; i < pts1.number_of_points(); ++i) {
                                cyclemap[i] = (float)cycle_vec.at(i);
                            }
                            Fmap branchmap = pts1.add_property_map<float>("scalar_Branch", 0).first;
                            for (int i = 0; i < pts1.number_of_points(); ++i) {
                                branchmap[i] = (float)branch_vec.at(i);
                            }
                            std::ofstream f("skel_points1234.ply");
                            f.precision(18);
                            f << pts1;
                        }
                        

                       // Nearest neighbor searching

                        Point_set pts2;
                        for (auto p : pc)
                            pts2.insert(p);
                        std::ofstream f1("poly_points.ply");
                        f1.precision(18);
                        f1 << pts2;

                        //std::vector<float> values;
                        values.reserve(input.number_of_points());
                        Pmap point_map2 = Pmap();
                        Neighborhood neighborhood(pc, point_map2);
                        int n = 1; // number of neighbours
                        KNeighborQuery neighbor_query = neighborhood.k_neighbor_query(n);
                        for (std::size_t i = 0; i < input.number_of_points(); ++i) {
                            std::vector<std::size_t> neighbors;
                            
                            const Point& point = get(input.point_map(), *(input.begin() + i));
                            neighbor_query(point, std::back_inserter(neighbors));
                            //Point& closest_point = get(point_map2, *(pc.begin() + neighbors[0]));
                            //float d = CGAL::sqrt((closest_point - point).squared_length());

                            size_t idx = neighbors[0];
                            //float d = cycle_vec.at(idx);
                            float d = branch_vec.at(idx);
                            values.push_back(d);
                        }
                   
                    
                    }

                    float value(std::size_t pt_index) {
                        //if (pt_index < values.size()) {
                            //std::cout << pt_index << "," << values.size() << std::endl;
                            return values[pt_index];
                        //}
                        //else {
                         //   return 0.0; 
                        //}
                    }



                    
            };
        
    }}}

