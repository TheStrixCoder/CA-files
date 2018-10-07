//16CO212 16CO249
//Computer Architecture Lab Assignment 0
//Question 1


#include <stdio.h>

//Printing Properties of the device

void printDevProp(cudaDeviceProp device_properties)
{
    printf("\tMajor revision number:         %d\n",  device_properties.major);
    
    printf("\tMinor revision number:         %d\n",  device_properties.minor);
    
    printf("\tName:                          %s\n",  device_properties.name);
   
    printf("\tTotal shared memory per block: %u\n",  device_properties.sharedMemPerBlock);

    printf("\tTotal global memory:           %u\n",  device_properties.totalGlobalMem);
    
    printf("\tTotal registers per block:     %d\n",  device_properties.regsPerBlock);
    
    printf("\tWarp size:                     %d\n",  device_properties.warpSize);
    
    printf("\tMaximum memory pitch:          %u\n",  device_properties.memPitch);
    
    printf("\tMaximum threads per block:     %d\n",  device_properties.maxThreadsPerBlock);
    
    for (int i = 0; i < 3; ++i)
              printf("\tMaximum dimension %d of block:  %d\n", i, device_properties.maxThreadsDim[i]);
    

    for (int i = 0; i < 3; ++i)
             printf("\tMaximum dimension %d of grid:   %d\n", i, device_properties.maxGridSize[i]);
    

    printf("\tClock rate:                    %d\n",  device_properties.clockRate);
   
    printf("\tTotal constant memory:         %u\n",  device_properties.totalConstMem);
   
    printf("\tTexture alignment:             %u\n",  device_properties.textureAlignment);
   
    printf("\tConcurrent copy and execution: %s\n",  (device_properties.deviceOverlap ? "Yes" : "No"));
   
    printf("\tNumber of multiprocessors:     %d\n",  device_properties.multiProcessorCount);
   
    printf("\tKernel execution timeout:      %s\n",  (device_properties.kernelExecTimeoutEnabled ? "Yes" : "No"));
   
    return;
}

int main()
{
    // variable to save number of CUDA devices
    int device_count;
    
    cudaGetDeviceCount(&device_count);
    
    printf("CUDA Device Query Code\n");
    
    printf("\tNumber of CUDA devices :%d .\n", device_count);

    // Iterate through all devices
    for (int i = 0; i < device_count; ++i)
    {
        // Get device properties

        printf("\nCUDA Device number %d\n", i);
        
        cudaDeviceProp device_properties;
      
        cudaGetDeviceProperties(&device_properties, i);
       
        printDevProp(device_properties);
    }

    return 0;
}