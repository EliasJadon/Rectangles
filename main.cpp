#include <igl/readOBJ.h>
#include <igl/internal_angles.h>
#include <igl/unproject_in_mesh.h>
#include <igl/unproject_ray.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

#include <Eigen/SVD>
#include <Eigen/Geometry>

#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>
#include <TinyAD/Utils/LineSearch.hh>

#include <igl/opengl/glfw/Viewer.h>
#include <math.h>
#include <mutex>
#include <thread>

#define BLUE_COLOR (Eigen::RowVector3d(0, 0, 1))
#define GREEN_COLOR (Eigen::RowVector3d(0, 1, 0))
#define RED_COLOR (Eigen::RowVector3d(1, 0, 0))

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

Eigen::MatrixXd V_2D_origin, V_2D, V_3D;
Eigen::MatrixXi F;
std::vector<Eigen::Matrix2d> rest_shapes;
Eigen::RowVector2d B2(0, 1);
Eigen::RowVector2d B1(1, 0);
igl::opengl::glfw::Viewer viewer;

std::vector<bool> is_vertex_pinned;
std::vector<bool> is_group_vertex;
std::vector<Eigen::RowVector2d> pin_coord;
std::vector<Eigen::RowVector3d> v_down_pos;

enum mouse_mode {PICK_SINGLE_VERTEX, PICK_GROUP_OF_VERTICES};
class mouse_params {
public:
	bool is_moving;
	int active_v_idx;
	int down_mouse_x, down_mouse_y;
	enum mouse_mode mode;

	mouse_params() {
		this->is_moving = false;
		this->active_v_idx = -1;
		this->mode = PICK_SINGLE_VERTEX;
	}
};

mouse_params mouse_p;
bool run_optimizer = false;
bool kill_optimizer_process = false;

double SD_weight = 0.01;
double RT_weight = 5;
double pin_vertices_weight = 100;
bool show_bounding_box = true;
bool show_rotate_per_face = true;
bool show_max_angle_per_face = false;
bool show_output_data = true;
int output_f = 0;

inline Eigen::MatrixXd from_2D_to_3D(const Eigen::MatrixXd& V_2D);
int get_vertex_from_mouse(Eigen::MatrixXd& V, Eigen::MatrixXi& F);
int get_face_from_mouse(Eigen::MatrixXd& V, Eigen::MatrixXi& F);
bool mouse_down(igl::opengl::glfw::Viewer& viewer, int button, int modifier);
bool mouse_up(igl::opengl::glfw::Viewer& viewer, int button, int modifier);
bool mouse_move(igl::opengl::glfw::Viewer& viewer, int mouse_x, int mouse_y);
bool pre_draw(igl::opengl::glfw::Viewer& viewer);
Eigen::RowVector3d computeTranslation(const int mouse_x, const int from_x, const int mouse_y, const int from_y, const Eigen::RowVector3d pt3D, igl::opengl::ViewerCore& core);

int main()
{
	std::string file_path = igl::file_dialog_open();
	if (file_path.length() == 0) {
		std::cerr << "Couldn't get file path!" << std::endl;
		return -1;
	}
	if (!igl::readOBJ(file_path, V_3D, F)) {
		std::cerr << "Could not open file!" << std::endl;
		return -1;
	}
	V_2D_origin = V_3D.leftCols(2);
	V_2D = V_2D_origin;

	is_vertex_pinned.resize(V_2D.rows());
	is_group_vertex.resize(V_2D.rows());
	for (int i = 0; i < is_vertex_pinned.size(); i++) {
		is_vertex_pinned[i] = false;
		is_group_vertex[i] = false;
	}
	pin_coord.resize(V_2D.rows());
	v_down_pos.resize(V_2D.rows());

	// Pre-compute triangle rest shapes in local coordinate systems
	rest_shapes.resize(F.rows());
	for (int f_idx = 0; f_idx < F.rows(); ++f_idx)
	{
		// Express a, b, c in local 2D coordiante system
		Eigen::Vector2d a = V_2D.row(F(f_idx, 0)).transpose();
		Eigen::Vector2d b = V_2D.row(F(f_idx, 1)).transpose();
		Eigen::Vector2d c = V_2D.row(F(f_idx, 2)).transpose();
		Eigen::Vector2d e1 = b - a;
		Eigen::Vector2d e2 = c - a;
		e1 = Eigen::Vector2d(e1.dot(B1), e1.dot(B2));
		e2 = Eigen::Vector2d(e2.dot(B1), e2.dot(B2));

		// Save 2-by-2 matrix with edge vectors as colums
		rest_shapes[f_idx] = TinyAD::col_mat(e1, e2);
	};

	viewer.data().set_mesh(from_2D_to_3D(V_2D), F);
	viewer.data().set_vertices(from_2D_to_3D(V_2D));
	viewer.data().compute_normals();

	// Set 2D screen mode (No rotation)
	viewer.core().trackball_angle = Eigen::Quaternionf::Identity();
	viewer.core().orthographic = true;
	viewer.core().set_rotation_type(igl::opengl::ViewerCore::RotationType::ROTATION_TYPE_NO_ROTATION);

	// Attach a menu plugin
	igl::opengl::glfw::imgui::ImGuiPlugin plugin;
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	plugin.widgets.push_back(&menu);
	viewer.plugins.push_back(&plugin);

	menu.callback_draw_viewer_menu = [&]()
	{
		if (ImGui::CollapsingHeader("Objectives weights", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::InputDouble("Symmetric Dirichlet", &SD_weight, 0, 0, "%.4f"))
			{
				SD_weight = std::max<double>(0, SD_weight);
			}
			if (ImGui::InputDouble("Rectangle", &RT_weight, 0, 0, "%.4f"))
			{
				RT_weight = std::max<double>(0, RT_weight);
			}
			if (ImGui::InputDouble("Pin vertices", &pin_vertices_weight, 0, 0, "%.4f"))
			{
				pin_vertices_weight = std::max<double>(0, pin_vertices_weight);
			}
		}
		ImGui::Checkbox("show_bounding_box", &show_bounding_box);
		ImGui::Checkbox("show_rotate_per_face", &show_rotate_per_face);
		ImGui::Checkbox("show_max_angle_per_face", &show_max_angle_per_face);
		ImGui::Checkbox("show_output_data", &show_output_data);
		// Draw parent menu content
		menu.draw_viewer_menu();
	};

	// Draw additional windows
	menu.callback_draw_custom_window = [&]()
	{
		if (output_f < 0 || !show_output_data) {
			return;
		}
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(500, 160), ImGuiCond_FirstUseEver);
		ImGui::Begin(std::string("Face " + std::to_string(output_f)).c_str(), nullptr, ImGuiWindowFlags_NoSavedSettings);
		
		Eigen::Vector2d a = V_2D.row(F(output_f, 0)).transpose();
		Eigen::Vector2d b = V_2D.row(F(output_f, 1)).transpose();
		Eigen::Vector2d c = V_2D.row(F(output_f, 2)).transpose();
		Eigen::Matrix2d M = TinyAD::col_mat(b - a, c - a);
		Eigen::Matrix2d Mr = rest_shapes[output_f];
		Eigen::Matrix2d J = M * Mr.inverse();
		Eigen::JacobiSVD<Eigen::Matrix2d> svd;
		svd.compute(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix2d U = svd.matrixU();
		Eigen::Matrix2d V = svd.matrixV();
		Eigen::Vector2d S = svd.singularValues();
		ImGui::Text("             J                    =                 U               *          S        *                V\n");
		ImGui::Text("(%9.4f, %9.4f) = (%9.4f, %9.4f) * (%9.4f) * (%9.4f, %9.4f)\n",
			J(0, 0), J(0, 1),
			U(0, 0), U(0, 1),
			S(0),
			V(0, 0), V(0, 1));
		ImGui::Text("(%9.4f, %9.4f)    (%9.4f, %9.4f)    (%9.4f)    (%9.4f, %9.4f)\n\n",
			J(1, 0), J(1, 1),
			U(1, 0), U(1, 1),
			S(1),
			V(1, 0), V(1, 1));
		
		Eigen::MatrixXd triangele_angles;
		igl::internal_angles(V_2D, F, triangele_angles);
		double triangle_max_angle = triangele_angles.row(output_f).maxCoeff() * 180.0 / M_PI;

		ImGui::Text("Biggest angle (degrees) = %.4f (== 90)\n", triangle_max_angle);
		{
			double a = J(0, 0);
			double b = J(0, 1);
			double c = J(1, 0);
			double d = J(1, 1);
			// y = 2ab + 2cd
			double y = 2 * a*b + 2 * c*d;
			// x = a^2 - b^2 + c^2 - d^2
			double x = std::pow(a, 2) - std::pow(b, 2) + std::pow(c, 2) - std::pow(d, 2);
			double theta_angle_radians = 0.5 * atan2(y, x);
			double theta_angle_degrees = theta_angle_radians * 180.0 / M_PI;
			ImGui::Text("V rotation angle (degrees) = %.4f (== {0,90,-90,180})\n", theta_angle_degrees);
			ImGui::Text("x = %.4f (!= 0)\n", x);
			ImGui::Text("y = %.4f (== 0)\n", y);
		}

		ImGui::End();
	};

	// Attach callback to allow toggling between 3D and 2D view
	viewer.callback_mouse_down = mouse_down;
	viewer.callback_mouse_up= mouse_up;
	viewer.callback_mouse_move = mouse_move;
	viewer.callback_pre_draw = pre_draw;
	viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer& viewer, unsigned int key, int /*mod*/)
	{
		if (key == '1')
		{
			// Switch between original & current model
			static int is_original_model = 0;
			viewer.data().set_vertices(is_original_model ? from_2D_to_3D(V_2D_origin) : from_2D_to_3D(V_2D));
			is_original_model = !is_original_model;
			return true;
		}
		if (key == '2') {
			if (mouse_p.mode == PICK_GROUP_OF_VERTICES) {
				mouse_p.mode = PICK_SINGLE_VERTEX;
			}
			else {
				mouse_p.mode = PICK_GROUP_OF_VERTICES;
			}	
		}
		if (key == ' ') 
		{	
			run_optimizer = !run_optimizer;
		}

		return false; // key press not used
	};


	std::thread optimization_thread(
		[&]()
		{
		// Set up function with 2D vertex positions as variables.
		auto func = TinyAD::scalar_function<2>(TinyAD::range(V_2D.rows()));

		// Add objective term per face. Each connecting 3 vertices.
		func.add_elements<3>(TinyAD::range(F.rows()), [&](auto& element)->TINYAD_SCALAR_TYPE(element)
		{
			// Evaluate element using either double or TinyAD::Double
			using T = TINYAD_SCALAR_TYPE(element);

			// Get variable 2D vertex positions
			Eigen::Index f_idx = element.handle;
			Eigen::Vector2<T> a = element.variables(F(f_idx, 0));
			Eigen::Vector2<T> b = element.variables(F(f_idx, 1));
			Eigen::Vector2<T> c = element.variables(F(f_idx, 2));

			// Triangle flipped?
			Eigen::Matrix2<T> M = TinyAD::col_mat(b - a, c - a);
			if (M.determinant() <= 0.0)
				return (T)INFINITY;

			// Get constant 2D rest shape of f
			Eigen::Matrix2d Mr = rest_shapes[f_idx];
			double A = 0.5 * Mr.determinant();

			// Compute symmetric Dirichlet energy
			Eigen::Matrix2<T> J = M * Mr.inverse();

			T result = 0;
			// Rectangle energy
			// A * (a*b+c*d)^2
			result += RT_weight * (A * (J(0, 0)*J(0, 1) + J(1, 0) * J(1, 1))*(J(0, 0)*J(0, 1) + J(1, 0) * J(1, 1)));

			// Symmetric Dirichlet energy
			result += SD_weight * A * (J.squaredNorm() + J.inverse().squaredNorm());

			return result;
		});

		// Add penalty term per constrained vertex.
		func.add_elements<1>(TinyAD::range(V_2D.rows()), [&](auto& element)->TINYAD_SCALAR_TYPE(element)
		{
			// Evaluate element using either double or TinyAD::Double
			using T = TINYAD_SCALAR_TYPE(element);
			if (!is_vertex_pinned[element.handle])
			{
				return 0;
			}
			Eigen::Vector2<T> p = element.variables(element.handle);
			Eigen::Vector2d p_target = pin_coord[element.handle].transpose();
			return pin_vertices_weight * (p_target - p).squaredNorm();
		});

		// Assemble inital x vector from P matrix.
		// x_from_data(...) takes a lambda function that maps
		// each variable handle (vertex index) to its initial 2D value (Eigen::Vector2d).
		Eigen::VectorXd x = func.x_from_data([&](int v_idx) { return V_2D.row(v_idx); });

		// Projected Newton
		TinyAD::LinearSolver solver;
		unsigned long i = 0;
		while (!kill_optimizer_process) {
			if(run_optimizer)
			{
				auto[f, g, H_proj] = func.eval_with_hessian_proj(x);
				TINYAD_DEBUG_OUT("Energy in iteration " << i++ << ": " << f);
				Eigen::VectorXd d = TinyAD::newton_direction(g, H_proj, solver);
				x = TinyAD::line_search(x, d, f, g, func);

				func.x_to_data(x, [&](int v_idx, const Eigen::Vector2d& p) {V_2D.row(v_idx) = p; });
				viewer.data().set_vertices(V_2D);
			}
		}
		});

	viewer.launch(/*resizable*/ true, /*fullscreen*/ false, "Rectangle POC App");
	kill_optimizer_process = true;
	if (optimization_thread.joinable())
	{
		optimization_thread.join();
	}

	return 0;
}

inline Eigen::MatrixXd from_2D_to_3D(const Eigen::MatrixXd& V_2D) {
	Eigen::MatrixXd V_3D;
	V_3D.resize(V_2D.rows(), 3);
	V_3D.setZero();
	V_3D.leftCols(2) = V_2D;
	return V_3D;
}

int get_vertex_from_mouse(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
	int vi = -1 /*Not found*/;

	// Cast a ray in the view direction starting from the mouse position
	double x = viewer.current_mouse_x;
	double y = viewer.core().viewport(3) - viewer.current_mouse_y;
	Eigen::Matrix<double, 3, 1, 0, 3, 1> pt;
	std::vector<igl::Hit> hits;
	unproject_in_mesh(
		Eigen::Vector2f(x, y),
		viewer.core().view,
		viewer.core().proj,
		viewer.core().viewport,
		V,
		F,
		pt,
		hits
	);
	if (hits.size() > 0)
	{
		int fi = hits[0].id;
		Eigen::RowVector3d bc;
		bc << 1.0 - hits[0].u - hits[0].v, hits[0].u, hits[0].v;
		bc.maxCoeff(&vi);
		vi = F(fi, vi);
	}
	return vi;
}

int get_face_from_mouse(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
	// Cast a ray in the view direction starting from the mouse position
	double x = viewer.current_mouse_x;
	double y = viewer.core().viewport(3) - viewer.current_mouse_y;
	Eigen::RowVector3d pt;
	Eigen::Matrix4f modelview = viewer.core().view;
	int vi = -1;
	std::vector<igl::Hit> hits;
	igl::unproject_in_mesh(Eigen::Vector2f(x, y), viewer.core().view, viewer.core().proj, viewer.core().viewport, V, F, pt, hits);
	Eigen::Vector3f s, dir;
	igl::unproject_ray(Eigen::Vector2f(x, y), viewer.core().view, viewer.core().proj, viewer.core().viewport, s, dir);
	int fi = -1;
	if (hits.size() > 0)
	{
		fi = hits[0].id;
	}
	return fi;
}

bool pre_draw(igl::opengl::glfw::Viewer& viewer) {
	{
		static bool only_once = true;
		if (only_once) {
			only_once = false;
			glfwMaximizeWindow(viewer.window);
		}
	}
	
	viewer.data().clear_points();
	viewer.data().clear_labels();
	viewer.data().clear_edges();
	// activate label rendering
	viewer.data().show_custom_labels = true;

	// Add pinned vertices
	for (int vi = 0; vi < is_vertex_pinned.size(); vi++) 
	{
		if (is_vertex_pinned[vi]) {
			Eigen::RowVector2d v_target = pin_coord[vi];
			Eigen::RowVector2d v_current = V_2D.row(vi);
			viewer.data().add_points(Eigen::RowVector3d(v_current(0), v_current(1), 0), (is_group_vertex[vi]) ? GREEN_COLOR : BLUE_COLOR);
			viewer.data().add_points(Eigen::RowVector3d(v_target(0), v_target(1), 0), RED_COLOR);
		}
	}

	// Add bounding box
	if(show_bounding_box)
	{
		static Eigen::Vector2d m = V_2D_origin.colwise().minCoeff();
		static Eigen::Vector2d M = V_2D_origin.colwise().maxCoeff();

		// Corners of the bounding box
		Eigen::MatrixXd V_box(4, 3);
		V_box <<
			m(0), m(1), 0,
			M(0), m(1), 0,
			M(0), M(1), 0,
			m(0), M(1), 0;

		// Edges of the bounding box
		Eigen::MatrixXi E_box(4, 2);
		E_box <<
			0, 1,
			1, 2,
			2, 3,
			3, 0;

		// Plot the corners of the bounding box as points
		viewer.data().add_points(V_box, RED_COLOR);

		// Plot the edges of the bounding box
		for (unsigned i = 0; i < E_box.rows(); ++i) {
			viewer.data().add_edges(
				V_box.row(E_box(i, 0)),
				V_box.row(E_box(i, 1)),
				RED_COLOR
			);
		}

		// Plot labels with the coordinates of bounding box vertices
		std::stringstream l1;
		l1 << "(" << m(0) << ", " << m(1) << ")";
		viewer.data().add_label(Eigen::Vector3d(m(0) - 0.1, m(1) - 0.1, 0), l1.str());
		std::stringstream l2;
		l2 << "(" << M(0) << ", " << M(1) << ")";
		viewer.data().add_label(Eigen::Vector3d(M(0) + 0.1, M(1) + 0.1, 0), l2.str());
	}

	if (show_max_angle_per_face || show_rotate_per_face) 
	{
		for (int f_idx = 0; f_idx < F.rows(); f_idx++) {
			Eigen::Vector2d a = V_2D.row(F(f_idx, 0)).transpose();
			Eigen::Vector2d b = V_2D.row(F(f_idx, 1)).transpose();
			Eigen::Vector2d c = V_2D.row(F(f_idx, 2)).transpose();
			Eigen::Vector2d label_pos = (a + b + c) / 3;
			Eigen::Matrix2d M = TinyAD::col_mat(b - a, c - a);
			Eigen::Matrix2d Mr = rest_shapes[f_idx];
			Eigen::Matrix2d J = M * Mr.inverse();
			Eigen::JacobiSVD<Eigen::Matrix2d> svd;
			svd.compute(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
			Eigen::Matrix2d U = svd.matrixU();
			Eigen::Matrix2d V = svd.matrixV();
			Eigen::Vector2d S = svd.singularValues();
			Eigen::MatrixXd triangele_angles;
			igl::internal_angles(V_2D, F, triangele_angles);
			double triangle_max_angle = triangele_angles.row(f_idx).maxCoeff() * 180.0 / M_PI;

			if (show_max_angle_per_face) 
			{
				viewer.data().add_label(Eigen::Vector3d(label_pos(0), label_pos(1), 0), std::to_string(triangle_max_angle));
			}
			if (show_rotate_per_face)
			{
				double a = J(0, 0);
				double b = J(0, 1);
				double c = J(1, 0);
				double d = J(1, 1);
				// y = 2ab + 2cd
				double y = 2 * a*b + 2 * c*d;
				// x = a^2 - b^2 + c^2 - d^2
				double x = std::pow(a, 2) - std::pow(b, 2) + std::pow(c, 2) - std::pow(d, 2);
				double theta_angle_radians = 0.5 * atan2(y, x);
				double theta_angle_degrees = theta_angle_radians * 180.0 / M_PI;
				viewer.data().add_label(Eigen::Vector3d(label_pos(0), label_pos(1), 0), std::to_string(theta_angle_degrees));
			}
		}
	}
	
	return false;
}

bool mouse_up(igl::opengl::glfw::Viewer& viewer, int /*button*/, int /*modifier*/)
{
	mouse_p.is_moving = false;
	return false;
}

bool mouse_down(igl::opengl::glfw::Viewer& viewer, int button, int /*modifier*/) 
{
	bool LeftClick = (button == GLFW_MOUSE_BUTTON_LEFT);
	bool RightClick = (button == GLFW_MOUSE_BUTTON_MIDDLE);

	int v_idx = get_vertex_from_mouse(from_2D_to_3D(V_2D), F);
	if (v_idx >= 0 /*a vertex is found*/)
	{
		if (RightClick) 
		{ 
			// Remove vertex
			is_vertex_pinned[v_idx] = false;
			is_group_vertex[v_idx] = false;
		}
		if (LeftClick) {
			// Add vertex
			mouse_p.is_moving = true;
			mouse_p.down_mouse_x = viewer.current_mouse_x;
			mouse_p.down_mouse_y = viewer.current_mouse_y;
			v_down_pos[v_idx] = Eigen::RowVector3d(V_2D(v_idx, 0), V_2D(v_idx, 1), 0);
			mouse_p.active_v_idx = v_idx;
			if (mouse_p.mode == PICK_GROUP_OF_VERTICES) {
				is_group_vertex[v_idx] = true;
			}
			is_vertex_pinned[v_idx] = true;
			pin_coord[v_idx] = V_2D.row(v_idx);
		}
		return true;
	}
	return false;
}

bool mouse_move(igl::opengl::glfw::Viewer& viewer, int mouse_x, int mouse_y)
{
	output_f = get_face_from_mouse(from_2D_to_3D(V_2D), F);
	if (mouse_p.is_moving) {
		Eigen::RowVector3d translation = computeTranslation(mouse_x, mouse_p.down_mouse_x, mouse_y, mouse_p.down_mouse_y, v_down_pos[mouse_p.active_v_idx], viewer.core());
		mouse_p.down_mouse_x = mouse_x;
		mouse_p.down_mouse_y = mouse_y;
		if (mouse_p.mode == PICK_SINGLE_VERTEX) {
			v_down_pos[mouse_p.active_v_idx] = v_down_pos[mouse_p.active_v_idx] + translation;
			pin_coord[mouse_p.active_v_idx] = Eigen::RowVector2d(v_down_pos[mouse_p.active_v_idx](0), v_down_pos[mouse_p.active_v_idx](1));
		}
		else if (mouse_p.mode == PICK_GROUP_OF_VERTICES) {
			for (int vi = 0; vi < V_2D.rows(); vi++) {
				if (is_group_vertex[vi]) {
					v_down_pos[vi] = v_down_pos[vi] + translation;
					pin_coord[vi] = Eigen::RowVector2d(v_down_pos[vi](0), v_down_pos[vi](1));
				}	
			}
		}
		return true;
	}
	return false;
}

Eigen::RowVector3d computeTranslation(
	const int mouse_x,
	const int from_x,
	const int mouse_y,
	const int from_y,
	const Eigen::RowVector3d pt3D,
	igl::opengl::ViewerCore& core)
{
	Eigen::Matrix4f modelview = core.view;
	//project the given point (typically the handle centroid) to get a screen space depth
	Eigen::Vector3f proj = igl::project(pt3D.transpose().cast<float>().eval(), modelview, core.proj, core.viewport);
	float depth = proj[2];
	double x, y;
	Eigen::Vector3f pos1, pos0;
	//unproject from- and to- points
	x = mouse_x;
	y = core.viewport(3) - mouse_y;
	pos1 = igl::unproject(Eigen::Vector3f(x, y, depth), modelview, core.proj, core.viewport);
	x = from_x;
	y = core.viewport(3) - from_y;
	pos0 = igl::unproject(Eigen::Vector3f(x, y, depth), modelview, core.proj, core.viewport);
	//translation is the vector connecting the two
	Eigen::Vector3f translation;
	translation = pos1 - pos0;
	return Eigen::RowVector3d(translation(0), translation(1), translation(2));
}

void update_perpendicular_axis(Eigen::RowVector2d& B2, Eigen::RowVector2d& B1)
{
	const double epsilon = 0.001;
	B2.normalize();

	if (abs(B2(0)) < epsilon) {
		if (B2(1) > 0) {
			B2 = Eigen::RowVector2d(0, 1);
			B1 = Eigen::RowVector2d(1, 0);
		}
		else {
			B2 = Eigen::RowVector2d(0, -1);
			B1 = Eigen::RowVector2d(-1, 0);
		}
		return;
	}
	
	// Calculate the slope of a vector in 2D
	double slope = B2(1) / B2(0);
	double perpendicular_slope = -(1 / slope);

	// Get the perpendicular axis
	B1 = Eigen::RowVector2d(1, perpendicular_slope);
	B1.normalize();
	return;
}
