import os
import matplotlib.pyplot as plt
import csv


def read_csv_file(file_path):
    data_list = {}
    
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            data_dict = {
                'x': [float(row['x1']), float(row['x2']), float(row['x3'])],
                'y': [float(row['y1']), float(row['y2']), float(row['y3'])]
            }
            data_list[row['id']] = data_dict
    
    return data_list

def get_next_point(triangles_points):
    xs = triangles_points.get('x')
    ys = triangles_points.get('y')

    if xs[0] == (xs[1]+xs[2])/2:
        return xs[0], ((ys[1]-ys[0])/(xs[1]-xs[0])*(xs[2]-xs[0])/2+ys[0])*2-ys[2]

    try:
        A = calc_A(xs, ys)
        B = calc_B(xs, ys)
    except ZeroDivisionError:
        xs = flip_triangle_order(xs)
        ys = flip_triangle_order(ys)
        A = calc_A(xs, ys)
        B = calc_B(xs, ys)

    x = 1/(A - B)*(A*xs[0]-ys[2]+ys[0]+B*(xs[2]-2*xs[0]))
    y = A*(x-xs[0]) + ys[0]
    return x, y

def rotate_triangle_order(triangles_points):
    xs = triangles_points.get('x')
    ys = triangles_points.get('y')
    xs = [xs[index] for index in [1,2,0]]
    ys = [ys[index] for index in [1,2,0]]
    return {'x': xs, 'y': ys}

def flip_triangle_order(xs):
    return [xs[index] for index in [0,2,1]]

def calc_B(xs, ys):
    return (ys[1]-ys[0])/(xs[1]-xs[0])

def calc_A(xs, ys):
    return (ys[1]+ys[2]-2*ys[0])/(xs[1]+xs[2]-2*xs[0])

class ClosedFigurePlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Closed Figure of Triangles')

    def add_figure(self, x_coords, y_coords):
        # Close the figure by appending the first point to the end
        x_coords_figure = x_coords + [x_coords[0]]
        y_coords_figure = y_coords + [y_coords[0]]

        # Plot the closed figure
        self.ax.plot(x_coords_figure, y_coords_figure, marker='o', linestyle='-', linewidth=2)

    def show_plot(self):
        plt.grid(True)
        plt.show()

class TriangleNode:
    def __init__(self, triangle, parent=None):
        self.parent_node = parent
        self.triangle = triangle
        self.childreen = []
    
    def generate_children(self, children_branch = 0):
        if children_branch % 3 == 0:
            x_1,y_1 = get_next_point(self.triangle)
            t_1 = self.get_vertice_for_triangle(x_1, y_1, 0, 1, 2)
            self.childreen.append(TriangleNode(t_1,self))
        if children_branch % 5 == 0:
            x_2,y_2 = get_next_point(rotate_triangle_order(self.triangle))
            t_2 = self.get_vertice_for_triangle(x_2, y_2, 1, 2, 0)
            self.childreen.append(TriangleNode(t_2,self))
        if children_branch % 7 == 0:
            x_3,y_3 = get_next_point(rotate_triangle_order(rotate_triangle_order(self.triangle)))
            t_3 = self.get_vertice_for_triangle(x_3, y_3, 2, 1, 0)
            self.childreen.append(TriangleNode(t_3,self))

    def get_vertice_for_triangle(self, x_1, y_1, index_0, index_1, index_2):
        d11 = (x_1-self.triangle['x'][index_1])**2 + (y_1-self.triangle['y'][index_1])**2
        d12 = (x_1-self.triangle['x'][index_2])**2 + (y_1-self.triangle['y'][index_2])**2
        if d11 > d12:
            x_choosen,y_choosen = self.triangle['x'][index_1],self.triangle['y'][index_1]
        else:
            x_choosen,y_choosen = self.triangle['x'][index_2],self.triangle['y'][index_2]
        t_1 = {'x':[x_1,self.triangle['x'][index_0],x_choosen],
               'y':[y_1,self.triangle['y'][index_0],y_choosen]}
               
        return t_1

    def generate_generations(self, generations=0, children_branch = 0):
        if generations == 0:
            return
        if len(self.childreen) == 0:
            self.generate_children(children_branch)
        for child in self.childreen:
            child.generate_generations(generations=generations-1, children_branch=children_branch)
    
    def get_children(self):
        return self.childreen

    def get_triangle_xs(self):
        return self.triangle['x']
    
    def get_triangle_ys(self):
        return self.triangle['y']
    
    def print_family_tree(self, level=0):
        print(f"Level: {level}")
        print(self.triangle)
        print('---')
        for child in self.childreen:
            child.print_family_tree(level + 1)
    
    def print_family_tree_area_sequence(self, level = 0):
        area_of_triangle = calculate_area_of_triangle(self.triangle)
        areas = {level: [area_of_triangle]}

        childreen_areas = {}
        for child in self.childreen:
            childreen_areas = merge_dicts_with_lists(child.print_family_tree_area_sequence(level + 1), childreen_areas)
        areas = merge_dicts_with_lists(areas, childreen_areas)
        if not self.parent_node:
            print(areas)
        return areas

    def get_family_tree_size(self):
        if self.parent_node is None:
            count_me = 1
        else:
            count_me = 0
        return count_me + len(self.childreen) + sum([child.get_family_tree_size() for child in self.childreen])

    def plot_family_tree(self, plotter:ClosedFigurePlotter, single_line=False):
        plotter.add_figure(self.get_triangle_xs(), self.get_triangle_ys())
        selected_childreen = self.get_children()
        if single_line:
            selected_childreen = [selected_childreen[-1]]
        for child in selected_childreen:
            child.plot_family_tree(plotter)

def calculate_area_of_triangle(coordinates):
    if 'x' not in coordinates or 'y' not in coordinates:
        raise ValueError("Invalid input dictionary. It must contain 'x' and 'y' keys.")

    x_coords = coordinates['x']
    y_coords = coordinates['y']

    if len(x_coords) != 3 or len(y_coords) != 3:
        raise ValueError("Invalid number of coordinates. A triangle must have 3 vertices.")

    area = 0.5 * abs(sum(x_coords[i] * y_coords[(i + 1) % 3] - x_coords[(i + 1) % 3] * y_coords[i] for i in range(3)))
    return area

def merge_dicts_with_lists(dict1, dict2):
    merged_dict = {}
    
    for key in set(dict1) | set(dict2):  # Union of keys from both dictionaries
        merged_dict[key] = dict1.get(key, []) + dict2.get(key, [])
    
    return merged_dict

def save_image(plotter, generation, triangle, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plotter.ax.set_xlabel('X')
    plotter.ax.set_ylabel('Y')
    plotter.ax.set_title(f'Triangle {triangle}: Generation {generation}')
    plotter.ax.grid(True)
    plotter.fig.savefig(os.path.join(output_dir, f'T{triangle}_G{generation}.png'))
    plt.close(plotter.fig)


if __name__ == '__main__':
    triangles = read_csv_file('/home/barbaruiva/Documents/dev/triangles_project/triangles.csv')
    for triangle in triangles:
        triangles_points = triangles.get(triangle)
        for generation in range(1,10,1):
            print(f'Generating generation {generation}...')
            root_node = TriangleNode(triangles_points)
            root_node.generate_generations(generation)
            print(f"{root_node.get_family_tree_size()} triangles on the family tree")
            
            output_directory = "/home/barbaruiva/Documents/dev/triangles_project/Images"
            plotter = ClosedFigurePlotter()  # Create a new plotter instance
            root_node.plot_family_tree(plotter)
            save_image(plotter, generation, triangle, output_directory)
