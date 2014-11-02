int nDevices;
cudaGetDeviceCount(&nDevices);
int = sharedMemPerBlock;

for (int i = 0; i < nDevices; i++)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    sharedMemPerBlock = prop.sharedMemPerBlock;
}
