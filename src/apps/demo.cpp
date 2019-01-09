/* sobfu includes a*/
#include <sobfu/sob_fusion.hpp>

/* boost includes */
#include <boost/program_options.hpp>

/* opencv includes */
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/* pcl includes */
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/visualization/pcl_visualizer.h>

/* sys headers */
#include <iostream>

/* vtk includes */
#include <vtkArrowSource.h>
#include <vtkCellArray.h>
#include <vtkGlyph2D.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkXMLImageDataWriter.h>

struct SobFuApp {
    SobFuApp(std::string file_path, std::string params_path, bool logger, bool visualizer, bool visualizer_detailed,
             bool verbose, bool vverbose)
        : exit_(false),
          file_path_(file_path),
          params_path_(params_path),
          logger_(logger),
          visualizer_(visualizer),
          visualizer_detailed_(visualizer_detailed),
          verbose_(verbose),
          vverbose_(vverbose) {
        /*
         * initialise parameters
         */

        Params params;

        if (verbose_) {
            params.verbosity = 1;
        } else if (vverbose_) {
            params.verbosity = 2;
        }

        /*
         * declare parameters to read from .ini
         */

        boost::program_options::options_description desc("parameters");
        declare_parameters(desc, params);

        /*
         * read parameters from .ini
         */

        boost::program_options::variables_map vm;
        read_parameters(desc, vm);

        /*
         * parse parameters stored in units of voxels
         */

        params.tsdf_trunc_dist = vm["TSDF_TRUNC_DIST"].as<float>() * params.voxel_sizes()[0];
        params.eta             = vm["ETA"].as<float>() * params.voxel_sizes()[0];
        params.volume_pose     = cv::Affine3f().translate(
            cv::Vec3f(-params.volume_size[0] / 2.f, -params.volume_size[1] / 2.f, vm["VOL_POSE_T_Z"].as<float>()));

        /*
         * initialise sobolevfusion
         */

        sobfu = std::make_shared<SobFusion>(params);
    }

    /*
     * declare sobfu parameters
     */

    void declare_parameters(boost::program_options::options_description &desc, Params &params) {
        /*
         * tsdf
         */

        desc.add_options()("VOL_DIMS_X", boost::program_options::value<int>(&params.volume_dims[0]),
                           "no. of voxels along x axis");
        desc.add_options()("VOL_DIMS_Y", boost::program_options::value<int>(&params.volume_dims[1]),
                           "no. of voxels along y axis");
        desc.add_options()("VOL_DIMS_Z", boost::program_options::value<int>(&params.volume_dims[2]),
                           "no. of voxels along z axis");

        desc.add_options()("VOL_SIZE_X", boost::program_options::value<float>(&params.volume_size[0]),
                           "vol. size along x axis (metres)");
        desc.add_options()("VOL_SIZE_Y", boost::program_options::value<float>(&params.volume_size[1]),
                           "vol. size along y axis (metres)");
        desc.add_options()("VOL_SIZE_Z", boost::program_options::value<float>(&params.volume_size[2]),
                           "vol. size along z axis (metres)");

        desc.add_options()("TSDF_TRUNC_DIST", boost::program_options::value<float>(), "truncation distance (voxels)");
        desc.add_options()("ETA", boost::program_options::value<float>(), "expected object thickness (voxels)");
        desc.add_options()("TSDF_MAX_WEIGHT", boost::program_options::value<float>(&params.tsdf_max_weight),
                           "max. tsdf weight");

        desc.add_options()("GRADIENT_DELTA_FACTOR", boost::program_options::value<float>(&params.gradient_delta_factor),
                           "delta factor when calculating tsdf gradient (voxels)");

        /*
         * camera
         */

        desc.add_options()("INTR_FX", boost::program_options::value<float>(&params.intr.fx), "focal length x");
        desc.add_options()("INTR_FY", boost::program_options::value<float>(&params.intr.fy), "focal length y");
        desc.add_options()("INTR_CX", boost::program_options::value<float>(&params.intr.cx), "principal point x");
        desc.add_options()("INTR_CY", boost::program_options::value<float>(&params.intr.cy), "principal point y");

        desc.add_options()("TRUNC_DEPTH", boost::program_options::value<float>(&params.icp_truncate_depth_dist),
                           "depth map truncation distance (metres)");
        desc.add_options()("VOL_POSE_T_Z", boost::program_options::value<float>(),
                           "camera to volume translation along z axis");

        /*
         * bilateral filter
         */

        desc.add_options()("BILATERAL_SIGMA_DEPTH", boost::program_options::value<float>(&params.bilateral_sigma_depth),
                           "bilateral filter sigma z");
        desc.add_options()("BILATERAL_SIGMA_SPATIAL",
                           boost::program_options::value<float>(&params.bilateral_sigma_spatial),
                           "bilateral filter sigma x-y");
        desc.add_options()("BILATERAL_KERNEL_SIZE", boost::program_options::value<int>(&params.bilateral_kernel_size),
                           "bilateral filter kernel size");

        /*
         * solver
         */
        desc.add_options()("START_FRAME", boost::program_options::value<int>(&params.start_frame),
                           "frame when to start registration");

        desc.add_options()("MAX_ITER", boost::program_options::value<int>(&params.max_iter),
                           "max. no. of iterations of the solver");
        desc.add_options()("MAX_UPDATE_NORM", boost::program_options::value<float>(&params.max_update_norm),
                           "max. update norm when running the solver");

        /* SOBOLEV */
        desc.add_options()("S", boost::program_options::value<int>(&params.s),
                           "Sobolev kernel size (currently only supports s=7");
        desc.add_options()("LAMBDA", boost::program_options::value<float>(&params.lambda),
                           "Sobolev filter parameter (currently only supports 0.05, 0.1, 0.2, and 0.4");

        /* FASTFUSION */
        desc.add_options()("ALPHA", boost::program_options::value<float>(&params.alpha), "gradient descent step size");
        desc.add_options()("W_REG", boost::program_options::value<float>(&params.w_reg), "regularisation weight");
    }

    /*
     * read parameters from params.ini
     */

    void read_parameters(boost::program_options::options_description &desc, boost::program_options::variables_map &vm) {
        std::ifstream settings_file(params_path_);

        boost::program_options::store(boost::program_options::parse_config_file(settings_file, desc), vm);
        boost::program_options::notify(vm);
    }

    /*
     * load colour and depth images
     */

    void load_files(std::vector<cv::String> *depths, std::vector<cv::String> *images, std::vector<cv::String> *masks) {
        if (!boost::filesystem::exists(file_path_)) {
            std::cerr << "error: directory '" << file_path_ << "' does not exist. exiting" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (!boost::filesystem::exists(file_path_ + "/depth") || !boost::filesystem::exists(file_path_ + "/color")) {
            std::cerr << "error: source directory should contain 'color' and 'depth' folders. exiting..." << std::endl;
            exit(EXIT_FAILURE);
        }

        cv::glob(file_path_ + "/depth", *depths);
        cv::glob(file_path_ + "/color", *images);

        std::sort((*depths).begin(), (*depths).end());
        std::sort((*images).begin(), (*images).end());

        if (boost::filesystem::exists(file_path_ + "/omask")) {
            cv::glob(file_path_ + "/omask", *masks);
            std::sort((*masks).begin(), (*masks).end());
        }
    }

    /*
     * create a new directory out if not already present inside the input folder
     */

    void create_output_directory() {
        out_path_ = file_path_ + "/meshes";
        boost::filesystem::path dir_meshes(out_path_);

        if (boost::filesystem::create_directory(dir_meshes)) {
            std::cout << "created output directory for meshes" << std::endl;
        }

        if (visualizer_ || visualizer_detailed_) {
            out_screenshot_path_ = file_path_ + "/screenshots";
            boost::filesystem::path dir_screenshots(out_screenshot_path_);

            if (boost::filesystem::create_directory(dir_screenshots)) {
                std::cout << "created output directory for screenshots" << std::endl;
            }
        }
    }

    /*
     * get no. of vertices in a mesh
     */

    int get_no_vertices(pcl::PolygonMesh::Ptr mesh) {
        pcl::PointCloud<pcl::PointXYZ> pc;
        pcl::fromPCLPointCloud2(mesh->cloud, pc);

        return pc.size();
    }

    /*
     * save output meshes in .vtk format
     */

    void save_mesh(pcl::PolygonMesh::Ptr mesh, int i, std::string name) {
        /* pad frame no. with 0's */
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << i;
        std::string frameNum = ss.str();

        /* save to vtk */
        pcl::io::saveVTKFile(out_path_ + "/" + name + "_" + frameNum + ".vtk", *mesh);
        std::cout << "saved " + name + "_" + frameNum + ".vtk" << std::endl;
    }

    /*
     * save 3d deformation field in .vtk format
     */

    void save_field(std::shared_ptr<SobFusion> sobfu, int i) {
        /* get parameters */
        Params params = sobfu->getParams();

        /* create vtk image */
        vtkSmartPointer<vtkImageData> image = vtkSmartPointer<vtkImageData>::New();
        /* specify size of image */
        image->SetDimensions(params.volume_dims[0], params.volume_dims[1], params.volume_dims[2]);

        /* specify size of image data */
        image->AllocateScalars(VTK_FLOAT, 4);
        int *dims = image->GetDimensions();

        /* copy vector field data */
        std::shared_ptr<sobfu::cuda::DeformationField> psi = sobfu->getDeformationField();
        kfusion::cuda::CudaData displacement_data          = psi->get_data();
        displacement_data.download(image->GetScalarPointer());

        /* pad frame no. with 0's */
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << i;
        std::string frameNum = ss.str();
        std::string fileName = out_path_ + "/field_" + frameNum + ".vti";

        /* save file */
        vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
        writer->SetFileName(fileName.c_str());
        writer->SetInputData(image);
        writer->Write();

        std::cout << "saved the vector field to .vti" << std::endl;
    }

    bool execute() {
        /* sobfu app */
        sobfu = sobfu;

        /* images */
        cv::Mat depth, image, mask;
        std::vector<cv::String> depths, images, masks;

        load_files(&depths, &images, &masks);

        /* output */
        create_output_directory();

        /* pipeline */
        double time_ms = 0;
        bool has_image = false;
        bool has_masks = masks.size() > 0;

        int v1(0);
        int v2(0);

        int v3(0);
        int v4(0);
        int v5(0);

        for (size_t i = 0; i < depths.size(); ++i) {
            depth = cv::imread(depths[i], CV_LOAD_IMAGE_ANYDEPTH);
            image = cv::imread(images[i], CV_LOAD_IMAGE_COLOR);

            cv::Mat depth_masked = cv::Mat::zeros(depth.size(), depth.type());
            if (has_masks) {
                mask = cv::imread(masks[i], CV_8U);
                depth.copyTo(depth_masked, mask);
            }

            if (!image.data || !depth.data) {
                std::cerr << "error: image could not be read; check for improper"
                          << " permissions or invalid formats. exiting..." << std::endl;
                exit(EXIT_FAILURE);
            }

            if (has_masks) {
                depth_device_.upload(depth_masked.data, depth_masked.step, depth_masked.rows, depth_masked.cols);
            } else {
                depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
            }

            {
                kfusion::SampledScopeTime fps(time_ms);
                has_image = (*(sobfu))(depth_device_);
            }

            /* get meshes */
            pcl::PolygonMesh::Ptr mesh_global, mesh_global_psi_inv;
            pcl::PolygonMesh::Ptr mesh_n, mesh_n_psi;

            if (visualizer_ || visualizer_detailed_ || logger_) {
                mesh_global               = sobfu->get_phi_global_mesh();
                int no_vertices_canonical = get_no_vertices(mesh_global);
                std::cout << "no. of point-normal pairs in the canonical model: " << no_vertices_canonical << std::endl;

                if (i >= 1) {
                    mesh_global_psi_inv                      = sobfu->get_phi_global_psi_inv_mesh();
                    int no_vertices_canonical_warped_to_live = get_no_vertices(mesh_global_psi_inv);
                    std::cout << "no. of point-normal pairs in the canonical model warped to live: "
                              << no_vertices_canonical_warped_to_live << std::endl;
                }

                if (visualizer_detailed_) {
                    if (i == 1) {
                        mesh_n     = sobfu->get_phi_n_mesh();
                        mesh_n_psi = sobfu->get_phi_n_psi_mesh();
                    }
                    if (i >= 2) {
                        mesh_n     = sobfu->get_phi_n_mesh();
                        mesh_n_psi = sobfu->get_phi_n_psi_mesh();
                    }
                }
            }

            if (logger_) {
                save_mesh(mesh_global, i, "canonical_mesh");
                if (i >= 1) {
                    save_mesh(mesh_global_psi_inv, i, "canonical_warped_to_live_mesh");
                }

                // save_field(sobfu, i);
            }

            if (visualizer_ || visualizer_detailed_) {
                if (i == 0) {
                    viewer = std::make_shared<pcl::visualization::PCLVisualizer>("meshes");
                }

                /*
                 * BASIC VISUALISER
                 *
                 */

                if (visualizer_) {
                    if (i == 0) {
                        /* create view ports */
                        viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
                        viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);

                        /* add labels */
                        viewer->addText("frame no. " + std::to_string(i), 20, 512, 15, 1.0, 1.0, 1.0, "frame_num", v1);

                        viewer->addText("canonical model", 15, 15, 15, 1.0, 1.0, 1.0, "v1 text", v1);
                        viewer->addText("canonical model warped to live", 15, 15, 15, 1.0, 1.0, 1.0, "v2 text", v2);

                        /* add bounding box */
                        cv::Vec3f vol_size    = sobfu->getParams().volume_size;
                        cv::Affine3f vol_pose = sobfu->getParams().volume_pose;

                        cv::Vec3f min = vol_pose * cv::Vec3f(0.f, 0.f, 0.f);
                        cv::Vec3f max = vol_pose * vol_size;

                        viewer->setCameraPosition(0.0, 0.0, max[2] + 3.0, 0.0, 0.0, 0.0);

                        /* display meshes */
                        viewer->addPolygonMesh(*mesh_global, "mesh canonical", v1);
                        viewer->spinOnce(5000);

                        /* save screenshot */
                        std::stringstream ss;
                        ss << std::setw(6) << std::setfill('0') << i;
                        std::string frame_num = ss.str();

                        viewer->saveScreenshot(out_screenshot_path_ + "/" + frame_num + ".png");
                    } else if (i >= 1) {
                        /* update label */
                        viewer->updateText("frame no. " + std::to_string(i), 20, 512, 15, 1.0, 1.0, 1.0, "frame_num");

                        /* display updated meshes */
                        viewer->updatePolygonMesh(*mesh_global, "mesh canonical");
                        if (i == 1) {
                            viewer->addPolygonMesh(*mesh_global_psi_inv, "mesh canonical warped to live", v2);
                        } else {
                            viewer->updatePolygonMesh(*mesh_global_psi_inv, "mesh canonical warped to live");
                        }
                        viewer->spinOnce(50);

                        /* save screenshot */
                        std::stringstream ss;
                        ss << std::setw(6) << std::setfill('0') << i;
                        std::string frame_num = ss.str();

                        viewer->saveScreenshot(out_screenshot_path_ + "/" + frame_num + ".png");
                    }
                }

                /*
                 * DETAILED VISUALISER
                 *
                 */

                else if (visualizer_detailed_) {
                    if (i == 0) {
                        /* create view ports */
                        viewer->createViewPort(0.0, 0.5, 0.5, 1.0, v1);
                        viewer->createViewPort(0.5, 0.5, 1.0, 1.0, v2);

                        viewer->createViewPort(0.0, 0.0, 0.5, 0.5, v3);
                        viewer->createViewPort(0.5, 0.0, 1.0, 0.5, v4);

                        /* add labels */
                        viewer->addText("frame no. " + std::to_string(i), 20, 244, 15, 1.0, 1.0, 1.0, "frame_num", v1);

                        viewer->addText("phi_n", 15, 15, 15, 1.0, 1.0, 1.0, "v1 text", v1);
                        viewer->addText("phi_global(psi_inv)", 15, 15, 15, 1.0, 1.0, 1.0, "v2 text", v2);

                        viewer->addText("phi_global", 15, 15, 15, 1.0, 1.0, 1.0, "v3 text", v3);
                        viewer->addText("phi_n(psi)", 15, 15, 15, 1.0, 1.0, 1.0, "v4 text", v4);

                        /* add bounding box */
                        cv::Vec3f vol_size    = sobfu->getParams().volume_size;
                        cv::Affine3f vol_pose = sobfu->getParams().volume_pose;

                        cv::Vec3f min = vol_pose * cv::Vec3f(0.f, 0.f, 0.f);
                        cv::Vec3f max = vol_pose * vol_size;

                        viewer->setCameraPosition(0.0, 0.0, max[2] + 3.0, 0.0, 0.0, 0.0);

                        /* display meshes */
                        viewer->addPolygonMesh(*mesh_global, "mesh canonical", v3);
                        viewer->spinOnce(10000);

                        /* save screenshot */
                        std::stringstream ss;
                        ss << std::setw(6) << std::setfill('0') << i;
                        std::string frame_num = ss.str();

                        viewer->saveScreenshot(out_screenshot_path_ + "/" + frame_num + ".png");
                    }
                    if (i >= 1) {
                        /* update labels */
                        viewer->updateText("frame no. " + std::to_string(i), 20, 244, 15, 1.0, 1.0, 1.0, "frame_num");
                        /* display meshes */
                        viewer->updatePolygonMesh(*mesh_global, "mesh canonical");

                        if (i == 1) {
                            viewer->addPolygonMesh(*mesh_n, "mesh n", v1);
                            viewer->addPolygonMesh(*mesh_global_psi_inv, "mesh canonical warped to live", v2);
                            viewer->addPolygonMesh(*mesh_n_psi, "mesh n psi", v4);
                        }
                        if (i >= 2) {
                            viewer->updatePolygonMesh(*mesh_n, "mesh n");
                            viewer->updatePolygonMesh(*mesh_global_psi_inv, "mesh canonical warped to live");
                            viewer->updatePolygonMesh(*mesh_n_psi, "mesh n psi");
                        }
                        viewer->spinOnce(50);

                        /* save screenshot */
                        std::stringstream ss;
                        ss << std::setw(6) << std::setfill('0') << i;
                        std::string frame_num = ss.str();

                        viewer->saveScreenshot(out_screenshot_path_ + "/" + frame_num + ".png");
                    }
                }
            }
        }

        return true;
    }

    kfusion::cuda::Depth depth_device_;

    std::shared_ptr<SobFusion> sobfu;
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

    bool exit_, logger_, visualizer_, visualizer_detailed_, off_screen_, verbose_, vverbose_;
    std::string file_path_, params_path_, out_path_, out_screenshot_path_;
};

/*
 * parse the input flag and determine the file path and whether or not to enable visualizer
 * all flags will be matched and the last argument which does not match the flag will be
 * treated as filepath
 */
void parse_flags(std::vector<std::string> args, std::string *file_path, std::string *params_path, bool *logger,
                 bool *visualizer, bool *visualizer_detailed, bool *verbose, bool *vverbose) {
    std::vector<std::string> flags = {"-h",           "--help",    "--enable-viz", "--enable-viz-detailed",
                                      "--enable-log", "--verbose", "--vverbose"};

    int idx = 0;
    for (auto arg : args) {
        if (std::find(std::begin(flags), std::end(flags), arg) != std::end(flags)) {
            if (arg == "-h" || arg == "--help") {
                std::cout << "USAGE: sobfu [OPTIONS] <file path> <ini path>" << std::endl;
                std::cout << "\t--help -h:    display help" << std::endl;
                std::cout << "\t--enable-viz: enable visualizer" << std::endl;
                std::cout << "\t--enable-viz-detailed: enable visualizer with additional meshes for debugging"
                          << std::endl;
                std::cout << "\t--enable-log: log output meshes" << std::endl;
                std::cout << "\t--verbose: low verbosity" << std::endl;
                std::cout << "\t--vverbose: high verbosity" << std::endl;
                std::exit(EXIT_SUCCESS);
            }

            if (arg == "--enable-log") {
                *logger = true;
            }
            if (arg == "--enable-viz") {
                *visualizer = true;
            }
            if (arg == "--enable-viz-detailed") {
                *visualizer_detailed = true;
            }
            if (arg == "--verbose") {
                *verbose = true;
            }
            if (arg == "--vverbose") {
                *vverbose = true;
            }
        } else if (idx == 0) {
            *file_path = arg;
            idx++;
        } else if (idx == 1) {
            *params_path = arg;
        }
    }
}

int main(int argc, char *argv[]) {
    int device = 0;

    kfusion::cuda::setDevice(device);
    kfusion::cuda::printShortCudaDeviceInfo(device);

    if (kfusion::cuda::checkIfPreFermiGPU(device)) {
        std::cout << std::endl
                  << "kinfu does not support pre-fermi gpu architectures, and is not built for them by "
                     "default; exiting..."
                  << std::endl;
        return 1;
    }

    /* program requires at least one argument--the path to the directory where the source files are */
    if (argc < 3) {
        return std::cerr << "error: incorrect number of arguments; please supply path to source data and .ini file; "
                            "exiting..."
                         << std::endl,
               -1;
    }

    std::vector<std::string> args(argv + 1, argv + argc);
    std::string file_path, params_path;

    bool logger              = false;
    bool visualizer          = false;
    bool visualizer_detailed = false;
    bool off_screen          = false;
    bool verbose             = false;
    bool vverbose            = false;

    parse_flags(args, &file_path, &params_path, &logger, &visualizer, &visualizer_detailed, &verbose, &vverbose);

    /* disable the visualiser when running over SSH */
    if (((visualizer || visualizer_detailed) && !off_screen) && getenv("SSH_CLIENT")) {
        return std::cerr << "error: cannot run visualiser while running over ssh; please run locally or disable the"
                            "visualiser. exiting..."
                         << std::endl,
               -1;
    }

    SobFuApp app(file_path, params_path, logger, visualizer, visualizer_detailed, verbose, vverbose);

    /* execute */
    app.execute();

    return 0;
}
