import "./modal.css";

export default function Modal({ open, close, data }) {
    // console.log("modal data : ", data);
    return (
        <div className={open ? 'openModal modal' : 'modal'}>
            {open ? (
                // 모달이 열릴때 openModal 클래스가 생성된다.
                <section className='h-[450px] bg-red-50'>

                    <main className='w-full p-4 flex flex-col items-center justify-start relative'>
                        <button className="absolute right-4 text-right text-[#999] text-xl" onClick={close}>
                            &times;
                        </button>
                        <div className='w-[450px] text-center text-2xl font-bold mt-4  mb-8 text-[#3b82f6]'>{data.data.label}</div>
                        <div id="map" className='w-[450px] h-[300px] bg-lime-100'>지도</div>

                    </main>
                    <footer className="pb-3 px-4">
                        <button className="w-[300px]" onClick={close}>
                            닫기
                        </button>
                    </footer>
                </section>
            ) : null}
        </div>
    )
}
