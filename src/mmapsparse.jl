using Base.Mmap

function mmapsave(spm::SparseMatrixCSC, fname::AbstractString)
    touch(fname)
    open(fname, "r+") do fhandle
        mmapsave(spm, fhandle)
    end
end

function mmapsave{Tv,Ti}(spm::SparseMatrixCSC{Tv,Ti}, fhandle::IO)
    header = Int64[spm.m, spm.n, length(spm.nzval), Base.Serializer.sertag(Tv), Base.Serializer.sertag(Ti)]

    seek(fhandle, 0)
    write(fhandle, reinterpret(UInt8, header))
    write(fhandle, reinterpret(UInt8, spm.colptr))
    write(fhandle, reinterpret(UInt8, spm.rowval))
    write(fhandle, reinterpret(UInt8, spm.nzval))
    nothing
end

function mmapload(fname::AbstractString)
    open(fname, "r+") do fhandle
        mmapload(fhandle)
    end
end

function mmapload(fhandle::IO)
    header = Array(Int64, 5)
    pos1 = position(fhandle)
    header = read!(fhandle, header)
    m = header[1]
    n = header[2]
    nz = header[3]
    Tv = Base.Serializer.desertag(Int32(header[4]))
    Ti = Base.Serializer.desertag(Int32(header[5]))

    pos1 += sizeof(header)
    colptr = Mmap.mmap(fhandle, Vector{Ti}, (n+1,), pos1)

    pos1 += sizeof(colptr)
    rowval = Mmap.mmap(fhandle, Vector{Ti}, (nz,), pos1)

    pos1 += sizeof(rowval)
    nzval = Mmap.mmap(fhandle, Vector{Tv}, (nz,), pos1)  
    SparseMatrixCSC{Tv,Ti}(m, n, colptr, rowval, nzval)
end 
