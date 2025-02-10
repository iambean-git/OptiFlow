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
        {" "}
        {/* 콘텐츠 오버레이 div 추가 */}
        <div className="flex flex-col">
          <img src="/images/logo_square_white.png" className="size-96 mb-16 animatedLogo" alt="logo" />
          {/* <section className="text-center">
              <p>
                  배수지의 수요 예측을 통해 심야 가동을 통해 전기료를 절약하고
              </p>
              <p>
                  주간에 필요한 만큼만 공급하여 전기료를 최적화합니다
              </p>
          </section> */}
          <span className="arrowSpan"></span>
        </div>
      </div>
    </div>
  );
}
