use cuda::driver;
use cuda::driver::Error as CudaError;
use cuda::driver::Direction;
use cuda::driver::{Context, Device, Function, Module};
use std::mem;
use std::ffi::CString;

pub struct DeviceBuffer<T> {
    size: usize,
    device_backing: *mut T,
}
impl<T> DeviceBuffer<T> {
    pub fn new(size: usize) -> Result<DeviceBuffer<T>, CudaError> {
        let bytes = size * mem::size_of::<T>();
        let device_backing = unsafe { driver::allocate(bytes)? as *mut T };
        Ok( DeviceBuffer{ size, device_backing } )
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn copy_to_device(&mut self, data: &[T]) -> Result<(), CudaError> {
        if self.size != data.len() {
            panic!("Device and host buffers must be the same size when copying to the device.");
        }
        unsafe {
            driver::copy(
                data.as_ptr(),
                self.device_backing,
                data.len(),
                Direction::HostToDevice,
            )
        }
    }

    pub fn copy_to_host(&self, data: &mut [T]) -> Result<(), CudaError> {
        if self.size != data.len() {
            panic!("Device and host buffers must be the same size when copying from the device.");
        }
        unsafe {
            driver::copy(
                self.device_backing as *const T,
                data.as_mut_ptr(),
                data.len(),
                Direction::DeviceToHost,
            )
        }
    }

    pub fn as_ptr(&self) -> *const T {
        self.device_backing as *const T
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.device_backing
    }
}
impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        unsafe { driver::deallocate(self.device_backing as *mut u8).unwrap() };
    }
}

pub struct CudaContext(Context);
lazy_static! {
    pub static ref STATIC_CONTEXT : CudaContext = CudaContext::new();
}
impl CudaContext {
    fn new() -> CudaContext {
        driver::initialize().expect("Failed to initialize CUDA driver API");
        let device = Device(0).expect("Failed to get CUDA device 0");
        let context = device.create_context().expect("Failed to create CUDA context");
        CudaContext(context)
    }
}
// This is probably unsound, but whatever...
unsafe impl Sync for CudaContext { }
unsafe impl Send for CudaContext { }
impl ::std::ops::Deref for CudaContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct CudaModule(Module<'static>);
lazy_static! {
    pub static ref MODULE : CudaModule = CudaModule::new(include_str!(env!("KERNEL_PTX_PATH")));
}
impl CudaModule {
    fn new(ptx_str: &str) -> CudaModule {
        let ptx = CString::new(ptx_str).expect("Unexpected failure converting PTX code to CString");

        let module = STATIC_CONTEXT.load_module(&ptx).expect("Failed to load module from PTX code.");
        CudaModule(module)
    }

    pub fn kernel<'a>(&'a self, name: &str) -> Result<Function<'static, 'a>, CudaError> {
        let kernel_name = CString::new(name).expect("Unable to convert string to CString");
        self.function(&kernel_name)
    }
}
unsafe impl Sync for CudaModule { }
unsafe impl Send for CudaModule { }
impl ::std::ops::Deref for CudaModule {
    type Target = Module<'static>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}