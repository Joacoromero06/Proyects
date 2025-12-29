<?php

function sumar($a, $b){
    return $a + $b;
}

$rtdo = sumar(3, 6);

echo "la suma es: " .$rtdo. "\n";

$day = 'Monday';
$month = 'January';

function calender():void{
    $day = 'Tuesday'; // local variable
    echo "local day: ".$day. "\n";

    $gday = $GLOBALS['day'];
    echo "global day: ".$gday. "\n";
    $gday = 0;

    global $month;
    echo "global month: ".$month. "\n";
    $month = 0;

}
calender();
echo "global day was changed: ".$gday. "\n"; // undefined
echo "global day was changed: ".$day. "\n"; // not changed

echo "global month was changed: ".$month. "\n"; // changed

?>