
export default function FetchFailed({msg}) {
    return (
        <div className='w-full h-full text-gray-600 flex flex-col justify-center items-center bg-white rounded-md'>
            <span className='text-lg'> {msg ? msg : ""} 데이터를 로드할 수 없습니다.</span>
            <button className='px-4 py-1 mt-4 border border-gray-400 rounded-lg text-sm'
                onClick={() => window.location.reload()}
            >
                다시 시도
            </button>

        </div>
    )
}
