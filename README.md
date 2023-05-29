# High-Performance Computing DEVS Simulator

The High-Performance Computing DEVS Simulator is a powerful tool designed for simulating Discrete Event Systems (DEVS) models in high-performance computing environments. Leveraging modern parallel computing architectures, including multi-core CPUs and GPUs, this simulator excels in accelerating simulations and handling large-scale, complex systems.

## Key Features

- **DEVS Modeling Framework:** Define and simulate complex systems using the flexible DEVS modeling framework.
- **High-Performance Computing:** Harness the power of multi-core CPUs and GPUs, utilizing parallel computing techniques to accelerate simulation execution and handle computationally intensive models.
- **Multi-GPU Support:** Efficiently distribute simulations across multiple GPUs, optimizing computational resource utilization for large-scale simulations.
- **Unified Memory Management:** Simplify memory allocation and data transfers between CPU and GPU with CUDA's unified memory management, enhancing performance and programmer productivity.
- **Scalability and Performance Optimization:** Incorporate advanced algorithms and optimization techniques to maximize simulation performance and scalability on high-performance computing architectures.

## Getting Started

To compile the High-Performance Computing DEVS Simulator using CMake, follow these instructions:

Make sure you have CMake installed on your system. You can download CMake from the official website (https://cmake.org/download/) and install it following the instructions for your operating system.

Create a new directory for the build files. Let's name it "build" for simplicity. Navigate to the root directory of the project in your terminal.

Inside the "build" directory, run the following command to generate the build files using CMake:


```shell
cmake ..
```

This command tells CMake to generate the build files based on the CMakeLists.txt file located in the parent directory (root directory).

After the build files are generated successfully, you can build the project using the appropriate build command for your platform. For example:

On Linux:

```shell
make
```

## Contributing

We welcome contributions from the community to enhance and improve the High-Performance Computing DEVS Simulator. If you're interested in contributing, please read our [Contribution Guidelines](link-to-contribution-guidelines).

## License

The High-Performance Computing DEVS Simulator is licensed under the [MIT License](link-to-license). See the [LICENSE](link-to-license) file for more details.

## Contact

For any inquiries or questions, please reach out to our team at [guillermotrabes@sce.carleton.ca](mailto:guillermotrabes@sce.carleton.ca).

Whether you're a researcher, engineer, or scientist involved in modeling and simulating complex systems, the High-Performance Computing DEVS Simulator provides a robust and efficient platform. With its focus on performance and scalability, it empowers users to tackle computationally demanding simulations and gain valuable insights into system dynamics. Join our community and explore the capabilities of this high-performance simulation tool for your modeling needs.
