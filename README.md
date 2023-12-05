# CellularAutomata
My numerous expeirments in CA

## abCA.py

![ca_evolution_20231120183918](https://github.com/EveryOneIsGross/CellularAutomata/assets/23621140/a48af7f3-7513-4711-a5b9-ade330a71788)


This project involves creating a unique approach to visualizing and exploring the Abramelin Ritual, a traditional ceremonial magic ritual, using Cellular Automata (CA). 

1. **Ritual Descriptions**: The project starts with a list of 17 steps describing the Abramelin Ritual, including creating a sacred space, performing daily prayers, and seeking guidance from a Holy Guardian Angel.

2. **Embedding and Similarity Calculation**: The text inputs, including the user's sentence related to the Abramelin Ritual, are converted into embeddings using the `Embed4All` model. Cosine similarity is then calculated between the user's input and each of the ritual descriptions.

3. **Cellular Automata (CA) Setup**: The CA grid is initialized with the ritual descriptions. The descriptions can be ranked based on similarity to the user's input, depending on user preference. This grid forms the basis for the CA evolution.

4. **CA Rule and Evolution**: The user provides a Wolfram CA rule number (between 0-255), grid size, and the number of CA steps. The CA rule determines how the grid evolves at each step.

5. **Visualization and Output**: Each step of the CA evolution is visualized and saved as an image. These images are then combined into a GIF to show the evolution over time. Additionally, a text file detailing the CA evolution and the corresponding ritual steps is generated.

6. **User Interaction**: The user interacts with the system by providing a sentence related to the Abramelin Ritual, choosing a CA rule number, specifying the grid size and the number of steps, and deciding whether to rank descriptions based on similarity.

The aim of the project is to provide a novel way to interpret and engage with the ritual sequences of the Abramelin Ritual through a computational and visual lens, blending traditional spiritual practices with modern computational techniques.

## automata_MAGICEYE.py

This project involves the creation of unique animated GIFs by combining Cellular Automata (CA) patterns with depth maps derived from existing GIF images. The key components and steps are as follows:

1. **Cellular Automaton Rule Generation**: A function `cellular_automaton_rule` converts a given rule number (between 0-255) into a binary string. This binary string represents the rule for the classic cellular automaton.

2. **CA Pattern Generation**: The `generate_ca_pattern` function creates a pattern based on the cellular automaton rule. It starts with a single active cell and evolves the pattern according to the rule.

3. **Brian's Brain Pattern Generation**: An alternative pattern generation method, `generate_brians_brain_pattern`, simulates a different cellular automaton known as Brian's Brain. This method initializes random cells and evolves them based on specific rules.

4. **Autostereogram Generation**: The `generate_autostereogram` function creates a single-image stereogram (Magic Eye image) using a depth map and a repeating pattern. This gives an illusion of 3D when viewed correctly.

5. **Depth Map Generation**: `generate_depth_map_from_image` converts an image frame from the GIF into a grayscale image, which is then used as a depth map for the autostereogram.

6. **Main Function and User Input**: In `main_updated`, the user inputs the CA rule number, column width for the rule pattern, and the path to the animated GIF. The user also chooses between Brian's Brain or classic cellular automaton for the pattern generation.

7. **Processing Frames**: The script reads frames from the provided GIF, applies the depth map and chosen pattern generation method to each frame, and then creates an autostereogram for each frame.

8. **Output**: The generated autostereograms are compiled into a new animated GIF, which is saved with a unique timestamp in the filename. 

9. **User Interaction**: Throughout the process, the user has control over the type of cellular automaton pattern, the rule number, and the source GIF, allowing for a variety of unique visual outputs.

The project creatively blends cellular automaton patterns with depth maps to produce intriguing 3D illusions in the form of animated GIFs, offering a novel way to visualize and interact with both CA patterns and existing imagery.

## bitmap2svq.py 

This project involves converting images into SVG (Scalable Vector Graphics) format with specific stylistic choices. The key components and steps are as follows:

1. **Image Conversion to Binary**: The function `convert_to_binary` reads an image from a specified path using OpenCV, converts it to grayscale, and optionally inverts it. The image can be resized based on a specified compression ratio. The image is then thresholded to create a binary image (black and white).

2. **SVG Creation**: The `save_svg` function takes the binary image and creates an SVG file. It iterates over each pixel in the image; if a pixel is white (value 255), it is represented in the SVG. The representation can be either a circle or a line (dot) at the pixel's position, based on the `use_circles_for_points` flag. The size of these points or circles can be specified.

3. **Processing Function**: `process_image_to_svg` is the main function that takes an input image path, output SVG path, threshold for binary conversion, point size for the SVG representation, and compression ratio. It combines the image conversion and SVG creation steps.

4. **User Interaction**: In the main block of the script, users can specify the input image path and output SVG path. The output SVG file is named with a timestamp to ensure uniqueness. The threshold, point size, and compression ratio can also be customized.

5. **Global Variables**: The script uses global variables for the compression ratio, whether to inverse the image, and whether to use circles or points for the SVG representation. These can be toggled to affect the output style.

6. **Output**: The output is an SVG file representing the original image in a stylized, vector format. This allows for scalability without loss of quality.

This project is particularly useful for creating stylized vector representations of images, offering customizable options such as image inversion, compression, and point representation style. It finds applications in areas where artistic or scalable representations of images are desired.

## caGEN_3D_XOR.py

![XOR3D](https://github.com/EveryOneIsGross/CellularAutomata/assets/23621140/65bbf893-42a4-4cc0-a432-728b3b4438fe)


This project involves generating and visualizing a 3D representation of cellular automata interactions using specific rules. The key components and steps are as follows:

1. **Rule Pattern Generation**: The function `generate_rule_pattern` creates a cellular automaton pattern based on a given rule number. This rule number is converted into a binary string representing various cell states (e.g., alive or dead) based on the neighboring cells.

2. **XOR Pattern Operation**: The function `xor_patterns` applies an XOR operation between two cellular automaton patterns, called the base pattern and the modulator pattern. This operation is used to create interaction effects between different patterns.

3. **3D Visualization Setup**: In the main function `main_3d`, the user inputs a carrier rule number and the size of the grid (n x n). Then, one or more modulator rule numbers are input.

4. **Layered Visualization in 3D**: For each modulator, the script generates its pattern, applies XOR with the carrier pattern, and visualizes the original modulator pattern, the carrier pattern, the reacted nodes (where the carrier and modulator patterns differ), and the XOR results in different colors and layers in a 3D plot. 

5. **Plotting and Output**: Each cellular automaton pattern and the results of their interactions are plotted in 3D space using Matplotlib's 3D projection capabilities. The carrier pattern is shown in green, modulator patterns in yellow, reacted nodes in red, and XOR results in blue. The visualization is saved as a high-resolution PNG file.

6. **Interactive Input and Iterative Process**: The user can input multiple modulators, and the carrier pattern is updated after each iteration, representing the cumulative interaction with all modulators.

7. **Visualization Aesthetics**: The 3D scatter plot includes customization for colors, markers, and axis labels, and the view is initialized with specific elevation and azimuth for better visual understanding.

This project showcases an innovative way to visualize complex interactions between multiple cellular automata patterns in a three-dimensional space, allowing for an in-depth analysis of pattern evolution and interactions over multiple generations.

## ca_analyse.py

This project is centered on analyzing the evolution of a cellular automaton over time, focusing on three key metrics: entropy, complexity, and fractal dimension. The main steps and components of the project are as follows:

1. **Grid Initialization**: A grid of size \(N \times N\) (with \(N = 100\)) is initialized with each cell randomly assigned to be alive (1) or dead (0) based on a probability \(p\) (set to 0.2). This grid serves as the starting point for the cellular automaton.

2. **Rule Definition**: A rule is defined for updating the grid. At each step, the state of each cell is updated based on the number of alive neighbors. The rules follow a variation of Conway's Game of Life: a cell dies if it has fewer than 2 or more than 3 alive neighbors, and a dead cell becomes alive if it has exactly 3 alive neighbors.

3. **Entropy Calculation**: The entropy of the grid is calculated at each generation. This measure quantifies the randomness or disorder within the grid.

4. **Complexity Calculation**: The complexity of the grid is computed by first obtaining the power spectrum through a Fast Fourier Transform (FFT) and then calculating the entropy of this spectrum. The complexity is a product of this entropy and the logarithm of the grid size, providing insight into the structural richness of the grid.

5. **Fractal Dimension Calculation**: The fractal dimension is calculated using a box-counting method. This measure gives an idea of how detailed the pattern is at different scales.

6. **Generational Evolution**: The grid is evolved over \(n = 64\) generations, with the rule applied iteratively. After each generation, the entropy, complexity, and fractal dimension are calculated and stored.

7. **Visualization**: The evolution of entropy, complexity, and fractal dimension over time is plotted in a series of graphs. Each metric is represented in a separate subplot, showing how these quantities change with each generation of the cellular automaton.

This project demonstrates a fascinating application of cellular automata, providing insights into the dynamic behavior of complex systems over time through quantitative measures. It offers a unique perspective on how simple rules can lead to complex and intricate patterns, with varying degrees of order, complexity, and scaling properties.

## gauss_laplace_ca.py

![ca_grid_gen_32](https://github.com/EveryOneIsGross/CellularAutomata/assets/23621140/802e2414-4fba-4f33-9e37-b52c81516b25)


This project involves simulating and visualizing the evolution of a cellular automaton (CA) with specific rules and parameters, and then generating GIFs to showcase the evolution over time. The key components and steps are:

1. **Grid Initialization**: The CA grid is initialized with a size of \(n \times n\) (with \(n = 512\)). The grid can be initialized either randomly or with a specific cluster pattern, based on the `init_type` variable.

2. **Cellular Automaton Evolution**: In each generation, the grid undergoes transformations based on Gaussian blurring and Laplacian filtering. The blurring kernel size increases with each generation.

3. **Dynamic State Updating**: The CA's state is updated based on the average values of the neighborhoods and predefined thresholds (`lower_threshold` and `upper_threshold`). The `allow_dying` flag determines whether cells can turn from alive to dead.

4. **Frame Saving**: For each generation, the CA state is visualized and saved as an individual frame. These frames are saved in a dedicated output folder, with filenames indicating the generation number.

5. **GIF Creation**: After all generations are processed, three different GIFs are created from the frames: one with all frames, one with odd-numbered frames, and one with even-numbered frames. These GIFs provide a dynamic visualization of the CA's evolution.

6. **Customization Options**: The script includes several global variables allowing for customization of the CA's behavior and the visualization, such as grid size, number of generations, initialization type, kernel size, and thresholds.

7. **Visualization and Output**: The project outputs GIFs that show the evolution of the cellular automaton over time. The visualizations can reveal complex patterns and behaviors emerging from simple rules.

In this project, the evolution of the cellular automaton (CA) is significantly influenced by image processing techniques, specifically Gaussian blurring and Laplacian filtering. These techniques contribute to the CA's dynamics in the following ways:

1. **Gaussian Blurring**:
   - **Purpose**: Gaussian blurring is used to smooth the grid at each generation. It essentially averages the values of a cell's neighbors within a certain radius (defined by the kernel size).
   - **Contribution to CA Evolution**:
     - **Neighborhood Averaging**: By averaging the states of neighboring cells, Gaussian blurring helps in creating a more gradual transition between alive and dead cells. This can lead to the emergence of new patterns that are less discrete and more fluid compared to traditional CA rules.
     - **Dynamic Kernel Size**: The kernel size of the Gaussian blur increases with each generation, meaning that as the CA evolves, the influence of a cell’s state on its distant neighbors becomes more pronounced. This leads to larger and more complex patterns as the CA progresses.

2. **Laplacian Filtering**:
   - **Purpose**: Laplacian filtering is an edge-detection method. It highlights regions of rapid intensity change in the CA grid, which are indicative of edges or boundaries between alive and dead cells.
   - **Contribution to CA Evolution**:
     - **Identifying Changes/Edges**: The Laplacian filter accentuates the boundaries or edges in the CA grid. This edge information is crucial for determining how patterns evolve, especially in determining where new cells should be born or existing cells should die.
     - **Feedback into State Update**: The output of the Laplacian filter, after normalization, is used to update the state of the CA. The decision to change a cell's state from alive to dead (or vice versa) depends on the average values in the neighborhood (after Laplacian filtering), compared against the predefined lower and upper thresholds.

3. **Integration with CA Rules**:
   - The results from the Gaussian blur and Laplacian filter are not used in isolation but are integrated into the CA's traditional rule set. For example, whether a cell lives, dies, or is born in the next generation is influenced by the filtered values from these processes.
   - The combination of these image processing techniques with traditional CA rules results in a unique and dynamic evolution pattern. The CA doesn't just rely on the immediate neighborhood state (as in classical rules) but also on the broader pattern context provided by the Gaussian and Laplacian filters.

4. **Dynamic and Complex Patterns**:
   - The use of these filters leads to a CA that can exhibit more complex and dynamic behaviors than traditional CA models. The patterns formed can be more nuanced, exhibiting a range of behaviors from smooth transitions to sudden emergent complexities, depending on the parameters set for the Gaussian blur and Laplacian filters.

In summary, Gaussian blurring and Laplacian filtering in this CA project serve to incorporate spatial context and edge information into the CA's evolution process. This integration allows for the development of more complex, dynamic, and visually interesting patterns than those typically seen in conventional cellular automata.

## gauss_laplace_goo_imageprocess.py

![me_pose_gen_255](https://github.com/EveryOneIsGross/CellularAutomata/assets/23621140/f0069164-6d9a-4c78-9d2b-bbbbe5ea52c5)

This project is centered on evolving a binary grid, which is initialized from an image, through a series of image processing steps over multiple generations, and then visualizing the evolution through GIFs. Here's a summary:

1. **Binary Grid Initialization**: The grid is initialized from an image. The image is converted to grayscale and then to a binary image using a threshold. This binary grid serves as the starting point for the cellular automaton (CA).

2. **Image Processing Techniques**:
   - **Gaussian Blurring**: Each generation of the CA undergoes Gaussian blurring. The kernel size of the blur increases with each generation, leading to a progressively smoother image over time.
   - **Laplacian Edge Detection**: Post-blurring, a Laplacian filter is applied for edge detection. This filter highlights regions of rapid intensity change, effectively capturing the boundaries or transitions in the grid.

3. **CA Evolution Step**:
   - The CA evolves through a combination of Gaussian blurring and Laplacian filtering, followed by a binary threshold operation. The result is a new binary state of the grid, representing the next generation of the CA.

4. **Saving Frames**:
   - For each generation, the state of the CA is captured as an image frame. These frames are saved as PNG files in an output folder, with each frame showing the CA state at a specific generation.

5. **GIF Creation**:
   - The frames are compiled into GIFs to visualize the CA's evolution over time. Three GIFs are created: one with all frames, one with only even-numbered frames, and one with only odd-numbered frames.

6. **User-Defined Variables**:
   - The script includes variables such as `kernel_size`, `CA_threshold`, and `num_generations` which can be adjusted to modify the behavior of the CA and the visualization.

7. **Output**:
   - The project outputs GIFs showcasing the dynamic evolution of the CA. Gooey.

This project demonstrates an interesting blend of cellular automata concepts with image processing techniques, leading to a unique and visually engaging representation of evolutionary patterns.

## gauss_laplace_imageprocess.py

Less gooey, more granular generation.

This project involves creating a cellular automaton (CA) that evolves over time, based on image processing techniques applied to an initial binary grid derived from an image. The key components and steps of the project are:

1. **Binary Grid Initialization**: The grid is initialized from an image, converted to grayscale, and then thresholded to create a binary grid (black and white pixels).

2. **Gaussian Blurring and Laplacian Filtering**:
   - **Gaussian Blurring**: In each generation, the binary grid is blurred using a Gaussian blur. The kernel size for blurring increases with each generation, which smoothens the grid more over time.
   - **Laplacian Filtering**: Post-blurring, a Laplacian filter is applied to the grid. This edge-detection technique highlights the boundaries or transitions within the grid.

3. **Evolution Step**:
   - The CA's state evolves by applying the Gaussian blur and Laplacian filter, followed by a thresholding operation. The thresholds (`lower_threshold` and `upper_threshold`) determine the conditions under which a cell in the grid stays alive, dies, or is born in the next generation.

4. **Visualization and Output**:
   - **Frame Saving**: The state of the CA is visualized and saved as an image frame for each generation. These frames are stored in an output folder.
   - **GIF Creation**: The frames are compiled into GIFs to visualize the CA's evolution. Separate GIFs are created for all frames, even-numbered frames, and odd-numbered frames.

5. **Customization Options**:
   - The script includes variables such as `kernel_size`, `input_threshold`, `num_generations`, and others, allowing for the customization of the CA's behavior and visualization.

6. **Allow Dying Option**:
   - The `allow_dying` boolean variable determines whether cells in the grid can transition from alive to dead based on the neighborhood average intensity after applying the Laplacian filter.

7. **Wrap Mode for Grid Edges**:
   - The `mode` variable allows for different handling of the grid edges. The default 'wrap' mode creates a toroidal (donut-shaped) grid where edges wrap around.

In summary, this project demonstrates a cellular automaton that evolves through image processing techniques, specifically Gaussian blurring and Laplacian filtering. The CA's behavior can be customized, and its evolution is visualized through GIFs, providing a dynamic and visually engaging representation of its generational changes.

## lapTRACER.py

This project combines cellular automata (CA), image processing, and audio synthesis to create complex states and generate playable audio files.

1. **Cellular Automata Generation**:
   - Two 1D cellular automata are created based on specified rules (`rule_1` and `rule_2`). The size of each CA is determined by `matrix_width` and `matrix_height`.
   - These CAs are modulated together using a bitwise XOR operation to create a complex state.

2. **Laplacian Filtering**:
   - The modulated CA is subjected to Laplacian filtering, a technique used for edge detection. This step highlights the transitions or boundaries in the CA's pattern.

3. **Dithering**:
   - The Laplacian-filtered CA undergoes dithering, a process that converts the image to a binary format based on a threshold, further enhancing the contrast in the pattern.

4. **Path Conversion**:
   - The processed CA is converted into a continuous path, which essentially traces the evolution of the pattern over time.

5. **Visualization**:
   - The dithered CA and the continuous path are visualized and saved as images. These visualizations represent the intricate patterns formed by the CA.

6. **Audio Synthesis**:
   - The Laplacian-filtered CA is also used to generate an audio waveform. This involves averaging the rows of the CA to create a 1D signal, normalizing it, and then stretching it to create a longer audio signal.
   - The signal is then converted into a 16-bit audio format and saved as a WAV file.

7. **Saving Output**:
   - Various outputs, including the visualizations and the audio file, are saved with unique timestamps to ensure that each run of the script produces distinct files.
   - Additionally, the path information is saved as a JSON file, which can also be visualized separately.

In summary, this project represents a fascinating blend of computational art and audio synthesis, where complex patterns generated by cellular automata are not only visualized in intricate images but also transformed into unique audio experiences. The use of Laplacian filtering and dithering adds an extra layer of complexity to the visual patterns, which in turn influences the audio output.

## laplaceMAZESOLVE.py

![maze_20231205_121027](https://github.com/EveryOneIsGross/CellularAutomata/assets/23621140/49403736-d384-4355-9458-00ea7570e163)

This project involves generating a complex maze, solving it using Laplace's equation, and then visualizing the solution. Here's a breakdown of the process:

1. **Maze Generation**:
   - A maze is generated with specified dimensions (`dim`), complexity, and density. The maze is represented as a grid where walls are marked as `1` and open spaces as `0`.
   - The complexity and density parameters control how intricate and densely packed the maze is. The maze's borders are also set as walls.

2. **Solving the Maze with Laplace's Equation**:
   - Laplace's equation is used to solve the maze. This involves assigning a high potential (e.g., 10) to the starting point and a low potential (e.g., 0) to the exit point.
   - The potential within the maze (`phi`) is iteratively updated based on the average of neighboring cell potentials, excluding the walls. The process continues until the potential distribution converges.

3. **Visualizing the Solution**:
   - The solution is visualized by plotting the maze and overlaying the gradient of the potential (`phi`). The gradient represents the path from high to low potential, effectively showing a solution path through the maze.
   - The visualization uses a streamplot to depict the flow from the start to the end of the maze, with arrows indicating the direction.

4. **Output**:
   - The visualization is saved as an image file with a timestamp in its filename to ensure uniqueness.

In summary, this project combines maze generation, numerical methods for solving Laplace's equation, and data visualization techniques. It demonstrates a creative approach to visualizing the solution of a maze, providing insights into how potential fields can guide pathfinding.

## ca_MODGRAVITY.py ca_MODEXPLODE.py

This project is a cellular automata, where the traditional CA patterns are not only modulated for complexity but also subjected to a gravity-like effect, resulting in a animation.

1. **Cellular Automaton Generation**:
   - The project starts by generating a cellular automaton pattern based on a given rule number (`carrier`). The rule is translated into a binary representation to determine the automaton's behavior.
   - The grid for the CA is initialized either with an optional initial condition or with a default setup where the middle cell of the top row is active.

2. **Modulation with Multiple Modulators**:
   - The primary CA pattern (`carrier`) can be modulated using one or more additional CAs (`modulators`). Each modulator rule number is used to generate a separate CA pattern.
   - These modulator patterns are combined with the carrier pattern using a bitwise XOR operation, resulting in a complex, modulated pattern.

3. **Gravity Effect**:
   - A unique aspect of this project is the application of a 'gravity' effect to the CA pattern. This simulates the behavior of black blocks 'falling' downwards in the grid, creating a dynamic evolution of the pattern over time.
   - The gravity animation is produced by iteratively moving the black blocks down until they either reach the bottom of the grid or are obstructed by another block.

4. **Animation and GIF Creation**:
   - The evolution of the CA pattern under the gravity effect is animated. Each frame of the animation captures the state of the CA at a specific step.
   - The frames are then compiled into an animated GIF, providing a visual representation of the CA's dynamic evolution under gravity.

5. **User Interaction and Output**:
   - The program prompts the user to input the carrier rule number, grid size, and modulator rule numbers. 
   - It then generates a CA pattern based on these inputs, applies the gravity effect, animates the process, and outputs the result as an animated GIF.
  

## ca_GUESS.py

Defines cellular automata (CA) for generating patterns, modulating them with other patterns, and attempting to predict the original pattern based on the modulated one. This project demonstrates a fascinating application of cellular automata in information encoding and decoding. It shows how patterns can be modulated and demodulated, and explores the effectiveness of different prediction strategies in recovering original information from modulated data.

1. **CA Pattern Generation**:
   - A cellular automaton pattern is generated based on a specified rule number (`carrier`). The CA is represented as a grid, where each cell's state is determined by its own state and the states of its two neighbors in the previous generation.

2. **Center Column Extraction**:
   - Only the center column of the CA grid is considered for further processing. This column represents the evolution of the CA over time, starting from a single active cell.

3. **Modulation with Other Rules**:
   - The center column is modulated using additional CAs generated from different rule numbers (`modulator_rules`). The modulation is performed using a bitwise XOR operation between the carrier column and each modulator column.

4. **Prediction Methods**:
   - Two methods are used to predict the original (carrier) column from the modulated column:
     - **Reversing Modulation**: This method attempts to recover the original column by applying the XOR operation again with the modulator columns in reverse order.
     - **Simple Replication**: A simpler prediction method that replicates the first few characters of the modulated column throughout its length.

5. **Saving Results and Analysis**:
   - The original, modulated, and predicted columns are saved to a file along with their prediction accuracies.
   - The prediction accuracy is calculated as the percentage of correctly predicted cells in the column compared to the original column.

6. **Displaying Results**:
   - The raw and modulated center columns, along with the predicted columns and their accuracies, are displayed on the console.



## ca_MODCOLORGUI.py

This project explores the generation and visualization of cellular automata (CA) patterns, their combination using bitwise XOR operations, and the creation of a Gradio interface for user interaction. Here's a summary:

1. **CA Pattern Generation**:
   - A function `generate_rule_pattern` creates a cellular automaton pattern based on a specified rule number. The rule number is translated into a binary format to determine the automaton's behavior.
   - The grid for the CA is initialized with a single active cell in the middle of the top row, and the pattern evolves down the grid.

2. **Combining CA Patterns with XOR and Colors**:
   - Multiple CA patterns are generated, each based on different rule numbers (the `carrier` and `modulators`).
   - These patterns are combined using a bitwise XOR operation. The result is then color-coded, assigning unique colors to each combined pattern for better visual differentiation.

3. **Visualization**:
   - The combined pattern is visualized using Matplotlib. The generated image uses a colormap to distinguish between the different patterns resulting from the XOR operation.

4. **Gradio Interface**:
   - A Gradio interface is set up to enable interactive pattern generation. Users can input the carrier rule number, grid size, and modulator rule numbers.
   - Upon submission, the interface calls the `generate_xor_colored_overlay` function to create and display the combined CA pattern.

5. **Temporary File Saving for Visualization**:
   - For visualization, the generated pattern is saved as a PNG file in a temporary location. This file is then used to display the output in the Gradio interface.

6. **Interactive Demonstration**:
   - The project demonstrates a user-friendly way to explore cellular automata combinations. The Gradio interface provides a simple and interactive platform for users to experiment with different rule numbers and observe the resulting patterns.

In summary, this project is a creative blend of cellular automata theory, image processing, and interactive web application development. It allows users to experiment with different rule sets for cellular automata, observe how they interact when combined with XOR operations, and view the results in a colorful and visually appealing format.

## ca_ENCODERLAB.py

![ca_lab](https://github.com/EveryOneIsGross/CellularAutomata/assets/23621140/468835b4-64c0-4b4b-a351-34092224749c)

This project involves creating an interactive interface for experimenting with cellular automata (CA), featuring various modulation techniques and the ability to encode and decode textual information. Here's a detailed summary:

1. **CA Pattern Generation**:
   - A function `generate_rule_pattern` is defined to create a CA pattern based on a specified rule number and an optional initial condition. The CA evolves in a grid where each cell's state depends on its own state and the states of its two neighbors in the previous generation.

2. **Modulation Techniques**:
   - The script supports various modulation methods to alter the original CA pattern:
     - **XOR Modulation**: Combines the carrier CA with one or more modulator CAs using a bitwise XOR operation.
     - **Shift Modulation**: Shifts the rows of the carrier CA based on the central values of the modulator CA.
     - **Frequency Modulation**: Applies Fourier Transform to modulate the frequency components of the carrier and modulator CAs.
     - **Multiplication Modulation**: Multiplies the carrier CA with the modulator CA(s).

3. **Encoding and Decoding Textual Information**:
   - The script can encode a user-provided sentence into a binary representation and use this as the initial condition for the CA. The top row of the CA pattern represents the encoded binary string.
   - A function `decode_top_row` is used to decode the binary string from the top row of the CA pattern back into text.

4. **Gradio Interface**:
   - A Gradio interface (`iface`) is created, allowing users to interactively experiment with different carrier rules, grid sizes, modulation types, and textual inputs.
   - The interface includes sliders, radio buttons, and text boxes for user input.

5. **Visualization and Output**:
   - The generated CA pattern is visualized and displayed in the interface. Additionally, the decoded text from the CA pattern is also shown.
   - The visualization includes options for different modulation types and the ability to encode a sentence into the CA pattern.

6. **Auxiliary Functions and Features**:
   - Functions like `calculate_entropy` and `calculate_lyapunov_exponent` are included to analyze the generated CA patterns.
   - Custom CSS is applied to the interface for a personalized look and feel, and a detailed description is provided to guide the users.

7. **Saving Results**:
   - The generated CA patterns and decoded messages are saved to files for record-keeping and further analysis.

In summary, this project creates a versatile and user-friendly platform for exploring cellular automata, offering advanced features like modulation and text encoding/decoding. It provides a unique tool for users to experiment with CA behavior, observe the effects of different modulation techniques, and understand the concept of information encoding within CA patterns.

## ca_DENSITYTEST.py

This project is an advanced exploration of cellular automata (CA), specifically focusing on pattern generation, modulation, and density analysis. It provides an interface for user input to customize the CA generation and analysis. Here's a detailed breakdown:

1. **CA Pattern Generation**:
   - The script defines a function to generate a CA pattern based on a given rule number. The rule number is converted into its binary representation to establish the CA's behavior.

2. **Customizable Pattern Size**:
   - The user can define the size of the CA grid (both height and width), allowing for the generation of patterns of various sizes.

3. **XOR Combination of CA Patterns**:
   - Two CA patterns, primarily based on Rule 30 and a user-defined modulating rule, are generated and then combined using a bitwise XOR operation. This results in a complex pattern that is a combination of the two individual patterns.

4. **Density Analysis**:
   - Functions are defined to compute and analyze the density of active cells (cells with a state of `1`) over generations. This involves calculating the density for each slice (column) of the pattern as well as for each generation.

5. **Visualization**:
   - The CA patterns and their density analyses are visualized using Matplotlib. This includes classic representations of the CA patterns, density plots, and bar graphs showing the density distribution across slices.

6. **User Input and Interactive Execution**:
   - The script includes a user-input function (`main_user_input`) where users can specify the number of generations and the modulating rule number. This allows for interactive experimentation with different CA rules and observing how they modulate Rule 30.

7. **Comparison of Densities**:
   - The script compares the density of active cells over generations between the original Rule 30 pattern and the XOR-combined pattern. This comparison is visualized to showcase the effect of modulation on the CA's behavior.

8. **Saving and Displaying Results**:
   - The CA patterns, along with their density analyses, are displayed in a series of plots, providing insights into the evolution and characteristics of the cellular automata.

In summary, this project offers a comprehensive and interactive tool for exploring cellular automata, particularly focusing on the behavior of Rule 30 and its interaction with other rules through XOR modulation. It emphasizes the analysis of pattern densities, providing visual insights into the complex dynamics of cellular automata.




