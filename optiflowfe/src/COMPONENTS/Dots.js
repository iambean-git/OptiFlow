const Dot = ({ num, currentPage }) => {
    return (
      <div
        className={`size-3 rounded-full border border-gray-300 duration-700
                  ${currentPage === num ? (num==3||num==4) ? "bg-gray-400" : "bg-white" :"bg-transparent"}
                  ${(currentPage==3||currentPage==4) ? "border-gray-400" : "border-gray-300" }`}
        style={{
          transition: "background-color 0.5s",
        }}
      ></div>
    );
  };
  
export default function Dots({currentPage}) {
  //메인페이지 스크롤 점
  return (
    <div className="fixed top-1/2 right-[3vw] z-10">
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          alignItems: "center",
          width: 20,
          height: 120,
        }}
      >
        <Dot num={1} currentPage={currentPage}></Dot>
        <Dot num={2} currentPage={currentPage}></Dot>
        <Dot num={3} currentPage={currentPage}></Dot>
        <Dot num={4} currentPage={currentPage}></Dot>
      </div>
    </div>
  )
}
