template <class T> class Array;
{
    T* data;

    inline T & opeartor() (int i, int j)
    {
        return data[i*ncols + j];  
    }
}

