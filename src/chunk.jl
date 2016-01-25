# Chunks are parts of a large data datastructure that can be:
#   - loaded on demand
#   - cached, but evicted on memory pressure
#   - optionally attached with locality for efficient IO (in future)
# A chunked files:
#   - has a metadata section that lists location and data range of each chunk
#   - chunk key type (data range) must have the `in`, `first`, `last`, and `length` methods defined

using LRUCache
using Base.Mmap

type MemMappedMatrix{T,N}
    val::Matrix{T}
end

type Chunk{K,V}
    path::AbstractString
    offset::Int
    size::Int
    keyrange::K
    valtype::Type{V}
    data::Nullable{WeakRef}
end

function Chunk(path::AbstractString, offset::Integer, size::Integer, keyrange, V)
    K = typeof(keyrange)
    Chunk{K,V}(path, offset, size, keyrange, V, Nullable{WeakRef}())
end

function Chunk(path::AbstractString, keyrange, V)
    size = filesize(path)
    K = typeof(keyrange)
    Chunk{K,V}(path, 0, size, keyrange, V, Nullable{WeakRef}())
end

function load{T,N}(::Type{MemMappedMatrix{T,N}}, chunk::Chunk)
    logmsg("loading memory mapping chunk $(chunk.path)")
    ncells = div(chunk.size, sizeof(T))
    M = Int(ncells/N)
    A = Mmap.mmap(chunk.path, Matrix{T}, (M,N), chunk.offset)
    MemMappedMatrix{T,N}(A)
end

function load{T<:Vector{UInt8}}(::Type{T}, chunk::Chunk)
    logmsg("loading chunk $(chunk.path)")
    open(chunk.path) do f
        seek(f, chunk.offset)
        databytes = Array(UInt8, chunk.size)
        read!(f, databytes)
        return databytes::T
    end
end

function load{T<:Matrix}(::Type{T}, chunk::Chunk)
    M = readcsv(load(Vector{UInt8}, chunk))
    M::T
end

function load{T<:SparseMatrixCSC}(::Type{T}, chunk::Chunk)
    A = load(Matrix{Float64}, chunk)
    rows = convert(Vector{Int64},   A[:,1]);
    cols = convert(Vector{Int64},   A[:,2]);
    vals = convert(Vector{Float64}, A[:,3]);

    # subtract keyrange to keep sparse matrix small
    cols .-= (first(chunk.keyrange) - 1)
    sparse(rows, cols, vals)::T
end

function data{K,V}(chunk::Chunk{K,V}, lrucache::LRU)
    if isnull(chunk.data)
        data = load(V, chunk)
        finalizer(data, x->finalize_chunk_data(chunk, x))
        chunk.data = WeakRef(data)
    end
    v = get(chunk.data).value
    lrucache[chunk.keyrange] = v
    v::V
end

function finalize_chunk_data(chunk::Chunk, data)
    #logmsg("unloading chunk $(chunk.path)")
    chunk.data = nothing
end

type ChunkedFile{K,V}
    keyrangetype::Type{K}
    valtype::Type{V}
    metapath::AbstractString
    chunks::Vector{Chunk{K,V}}
    lrucache::LRU
end

function ChunkedFile(metapath::AbstractString, K, V, max_cache)
    chunks = Chunk{K,V}[]
    meta = (filesize(metapath) == 0) ? Array(Any,0,0) : readcsv(metapath)
    for idx in 1:size(meta,1)
        fname = meta[idx,3]
        push!(chunks, Chunk(fname, 0, filesize(fname), Int(meta[idx,1]):Int(meta[idx,2]), V))
    end
    ChunkedFile{K,V}(K, V, metapath, chunks, LRU(max_cache))
end

function writemeta(cf::ChunkedFile)
    chunkpfx = splitext(cf.metapath)[1]
    open(cf.metapath, "w") do meta
        chunks = cf.chunks
        idx = 1
        for chunk in chunks
            println(meta, first(chunk.keyrange), ",", last(chunk.keyrange), ",", chunkpfx, ".", idx)
            idx += 1
        end
    end
end

ChunkedSparseMatrixCSC(metapath::AbstractString, max_cache::Int=5) = ChunkedFile(metapath, UnitRange{Int64}, SparseMatrixCSC{Float64,Int}, max_cache)
ChunkedMemMappedMatrix(metapath::AbstractString, ncols::Int, max_cache::Int=5) = ChunkedFile(metapath, UnitRange{Int64}, MemMappedMatrix{Float64,ncols}, max_cache)

function getchunk{K,V}(cf::ChunkedFile{K,V}, key::Int)
    for chunk in cf.chunks
        (key in chunk.keyrange) && return chunk
    end
    error("Key not found")
end

function data{K,SK,SV}(cf::ChunkedFile{K,SparseMatrixCSC{SK,SV}}, key::Int)
    chunk = getchunk(cf, key)
    d = data(chunk, cf.lrucache)
    RecSys._sprowsvals(d, key - first(chunk.keyrange) + 1)
end

function data{K,T,N}(cf::ChunkedFile{K,MemMappedMatrix{T,N}}, key::Int)
    M = data(getchunk(cf, key), cf.lrucache)
    M.val
end