import NavBar from "../components/NavBar";

export default function Sample() {
    return (
        <div className="w-full min-w-[1000px] h-screen bg-[#f2f2f2] ">
            <NavBar />
            <div className="w-full h-screen pl-[260px] flex flex-col">
                <section className="w-full h-[160px] px-10 flex justify-between bg-red-50">
                </section>

                <section className="px-10 pb-10 pt-6 w-full h-full">
                    <div className="w-full h-full border rounded-lg ">
                    </div>
                </section>
            </div>
        </div>
    )
}

