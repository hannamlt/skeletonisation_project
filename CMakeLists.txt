cmake_minimum_required(VERSION 3.1...3.23)

project(Skeleton_project)

find_package(CGAL REQUIRED)
find_package(Eigen3 3.2.0)
include(CGAL_Eigen3_support)

create_single_source_cgal_program("skeletonise.cpp")
target_link_libraries(skeletonise PUBLIC CGAL::Eigen3_support)

create_single_source_cgal_program("skeletonise2.cpp")
target_link_libraries(skeletonise2 PUBLIC CGAL::Eigen3_support)

create_single_source_cgal_program("skeletonise3.cpp")
target_link_libraries(skeletonise3 PUBLIC CGAL::Eigen3_support)

create_single_source_cgal_program("skeletonise4.cpp")
target_link_libraries(skeletonise4 PUBLIC CGAL::Eigen3_support)

create_single_source_cgal_program("skeletonise_cycles.cpp")
target_link_libraries(skeletonise_cycles PUBLIC CGAL::Eigen3_support)


create_single_source_cgal_program("skeletonise_branches.cpp")
target_link_libraries(skeletonise_branches PUBLIC CGAL::Eigen3_support)

find_package(Boost OPTIONAL_COMPONENTS serialization iostreams)
include(CGAL_Boost_serialization_support)
include(CGAL_Boost_iostreams_support)

set(Boost_BZIP2_LIBRARY_DEBUG "C:/dev/vcpkg/installed/x64-windows/debug/lib/bz2d.lib")
set(Boost_ZLIB_LIBRARY_DEBUG "C:/dev/vcpkg/installed/x64-windows/debug/lib/zlibd.lib")
set(Boost_BZIP2_LIBRARY_RELEASE "C:/dev/vcpkg/installed/x64-windows/lib/bz2.lib")
set(Boost_ZLIB_LIBRARY_RELEASE "C:/dev/vcpkg/installed/x64-windows/lib/zlib.lib")

set(Boost_BZIP2_LIBRARY_DEBUG "C:/dev/vcpkg/installed/x64-windows/debug/lib/bz2d.lib")
set(Boost_ZLIB_LIBRARY_DEBUG "C:/dev/vcpkg/installed/x64-windows/debug/lib/zlibd.lib")
set(Boost_BZIP2_LIBRARY_RELEASE "C:/dev/vcpkg/installed/x64-windows/lib/bz2.lib")
set(Boost_ZLIB_LIBRARY_RELEASE "C:/dev/vcpkg/installed/x64-windows/lib/zlib.lib")


create_single_source_cgal_program("skeletonise_classifier.cpp")
target_link_libraries(skeletonise_classifier PUBLIC CGAL::Eigen3_support)