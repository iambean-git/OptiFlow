import "../../css/videoStyle.css"; 

export default function MainComponent1() {
  return (
    <div className="main-component-container">
      {" "}
      {/* 컨테이너 div 추가 */}
      <video
        src="/videos/waterfall.mp4"
        loop
        muted
        autoPlay
        className="background-video" 
      />
      <div className="content-overlay">
        <div className="flex flex-col">
          <img src="/images/logo_square_white.png" className="size-96 mb-16 animatedLogo" alt="logo" />
          <span className="arrowSpan"></span>
        </div>
      </div>
    </div>
  );
}
