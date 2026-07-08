import React from "react"

function sumar(x: number, y: number){
  return x + y
}


// Union types
// type color = "red" | "blue" | "green" | "yellow"

type LinkProps = React.ComponentProps<"a">


function Link({ target }: LinkProps){

  return (
    <a target={target}>test</a>
  )
}

export default function Page(){
  return (
    <div>
      <Link
        target="_blank"
      />
    </div>
  )
}