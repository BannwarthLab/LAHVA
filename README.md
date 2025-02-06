![LAHVA-Logo](./graphics/lahva_logo.jpg)

# LAHVA - Linear Algebra on Heterogenous/Vectorized Architecture

## Motivation & Purpose

The motivation behind the LAHVA project is to create a commodity layer that enables faster and more user-friendly interaction with heterogeneous computing hardware. The API of BLAS and LAPACK libraries are usually cumbersome especially when moving to accelerator hardware such as GPUs. Due to the complexity of the hardware and the communication between host and device (the accelerator) more objects are needed to control the execution of linear algebra operations. We want to come in and simplify the API by bundling the additional objects needed for the execution in a runtime. Additionally, we have implemented Tensor classes for Vector, Matrix and Lower triangular matrix. Due to the fact that memory spaces of the host and device are usually separate, we also need to take care of the transfer of information and addressing the right memory space in functions. Therefore, we went for a solution where a Tensor object can have to pointers, a host and device pointer. Allocators can then be used for both to allocate the memory. For GPU allocators we also implement the transfer within the allocator object.
The project is heavily focussed on using template variables for numeric precision as well as execution of the function on either host or device merely by changing the used Runtime. 

The iceberg symbolizes graphically the motivation of this project to simplify the interface between LAHVA and a vendor BLAS library such as nvidia's cuBLAS.

![Graphical Motivation](./graphics/iceberg.png)


## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

### Setup Tensor classes

## Compatibility 

We test the implementation for a permutation of the following operating systems, compilers and BLAS/LAPACK implementations:

%ToDo to a table with said stuff

## Support

Please open an issue in this GitLab repo, so we can help you out.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Authors and acknowledgment

Original author: Pit Steinbach  
with contributions from: Mark Heezen  
under the supervision of: Christoph Bannwarth

## License
For open source projects, say how it is licensed.

## Project status

This project is still in an experimental state, though we are committed to keep the API stable, changes could occur.
