import NavBar from "../components/NavBar";

export default function Dashboard() {

  return (
    <div className="w-full min-w-[1000px] h-screen bg-[#f2f2f2] ">
      <NavBar />
      <div className="w-full h-screen pl-[260px] flex flex-col">
        <div className="w-full h-[160px] px-10 flex justify-between">
          {/* 텍스트 */}
          <div className="w-2/5 h-full  flex flex-col justify-end text-[#333333]">
            <h1 className="text-4xl ">타이틀</h1>
            <p className="mt-2">각 배수지를 클릭하면, <span className="whitespace-nowrap"> 세부 정보를 확인할 수 있습니다.</span></p>
          </div>

          {/* 달력 */}
          <div className="h-full relative min-w-72 ">
            selectbox
          </div>
        </div>
        <section className="px-10 pb-10 pt-6 w-full h-full">
          <div className="w-full h-full rounded-lg flex flex-col">

            <div className="h-1/2 w-full flex gap-4">
              <section className="w-2/3 bg-white rounded-lg">
                영역1
              </section>

              <section className="w-1/3 bg-white rounded-lg">
                영역2
              </section>
            </div>

            <div className="h-1/2 w-full flex pt-4 gap-4">
              <section className="w-1/2 bg-white rounded-lg">
                영역1
              </section>

              <section className="w-1/2 bg-white rounded-lg">
                영역2
              </section>
            </div>





          </div>
        </section>
      </div>
    </div>
  );
}
