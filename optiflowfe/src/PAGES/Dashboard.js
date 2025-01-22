import NavBar from "../COMPONENTS/NavBar";
import WaterFlow from "../COMPONENTS/waterFlow/WaterFlow";

export default function Dashboard() {
  return (
    <div className="w-full h-screen ">
      <NavBar/>
      <div className="w-full h-screen pl-[260px] flex flex-col"> 
        <div className="w-full h-[160px]">
          selection 영역
        </div>
        <section className="p-10 w-full h-full">
          <div className="w-full h-full border rounded-lg ">
            <WaterFlow />
          </div>
        </section>
      </div>
    </div>
  );
}
