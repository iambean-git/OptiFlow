const Dot = ({ num, currentPage }) => {
    return (
      <div
        className="size-3 rounded-full border border-gray-300 duration-700"
        style={{
          backgroundColor: currentPage === num ? "gray" : "transparent",
          transition: "background-color 0.5s",
        }}
      ></div>
    );
  };
  
export default function Dots({currentPage}) {
  //메인페이지 스크롤 점
  return (
    <div style={{ position: "fixed", top: "50%", right: "5vw" }}>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          alignItems: "center",
          width: 20,
          height: 100,
        }}
      >
        <Dot num={1} currentPage={currentPage}></Dot>
        <Dot num={2} currentPage={currentPage}></Dot>
        <Dot num={3} currentPage={currentPage}></Dot>
      </div>
    </div>
  )
}
