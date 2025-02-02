
export default function MainComponent2() {
  return (
    <div className="inner scorlldown ">
        <section className="w-2/5 h-full bg-slate-50 flex flex-col justify-center px-16">
          <p className="text-[#1D5673] font-semibold text">About OptiFlow</p>

          <div className="text-2xl mt-3 font-semibold">
            <p>최소한의 비용으로</p>
            <p>효율적인 용수 공급을 위한 시스템을 제공합니다</p>
          </div>
          
          <div className="mt-10">
            <p>주간 전기 요금 00원</p>
            <p>야간 전기 요금 00원</p>
          </div>
          
        </section>
        <div className="w-3/5 bg-red-50">

        </div>
        <span className="arrowSpan"></span>
    </div>
  )
}
